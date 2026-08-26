"""Run a MedAgentGym experiment across parallel Daytona sandboxes.

Three problems this solves that a local run does not:

* **Isolation.** `validate_code` runs agent-generated code with the harness's own
  interpreter and `cwd`, so a local run leaves stray files in the repo and lets
  the agent `pip install` into the venv it is being graded in. A sandbox is
  thrown away afterwards, so every rollout starts from the same clean tree.
* **Docker.** MedAgentBench needs the HAPI FHIR server that upstream starts with
  `docker run -p 8080:8080`. Here every shard gets one (built from
  `Dockerfile.fhir`), reached over its preview URL, so no Docker daemon is
  needed anywhere and no shard sees another shard's writes -- `--with-fhir`.
* **Throughput.** Task indices are sharded across sandboxes, each running its
  shard with joblib, so total concurrency is `--sandboxes` x `--n-jobs`. Rate
  limits, not CPU, are usually what caps this.

The repo is cloned into each sandbox at a pinned ref rather than uploaded, so a
run is reproducible from a commit and nothing local leaks into the image.

Credentials are read from the local `credentials.toml` and written into each
sandbox, because that file is how the harness loads them. They therefore leave
this machine: use keys scoped to this work. `--with-fhir` additionally marks the
FHIR sandboxes public, since the worker sandboxes have to reach them and hold no
Daytona token of their own.

Usage:

    export DAYTONA_API_KEY=...
    uv run --extra daytona python scripts/daytona_run.py \\
        --task biocoder --sandboxes 4 --n-jobs 2

    # the same ten-per-dataset sample the local smoke runs use
    uv run --extra daytona python scripts/daytona_run.py \\
        --task medcalcbench --indices-path data/smoke_indices.json --sandboxes 2

    # MedAgentBench, with the FHIR server in a sandbox of its own
    uv run --extra daytona python scripts/daytona_run.py \\
        --task medagentbench --with-fhir --sandboxes 2
"""

import argparse
import inspect
import io
import json
import os
import re
import shlex
import sys
import tarfile
import textwrap
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import toml
from daytona import (
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    Daytona,
    DaytonaConfig,
    DaytonaError,
    Image,
    Resources,
)

REPO_DIR = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/home/MedAgentGym"
# Prebuilt by Dockerfile.daytona, one level above the clone so that `uv sync` in
# the clone finds a warm environment instead of resolving the `tasks` extra from
# scratch in every sandbox.
REMOTE_VENV = "/home/.venv"
DEFAULT_REPO_URL = "https://github.com/lmnr-ai/MedAgentGym.git"
FHIR_PORT = 8080
# How a shard reports itself to a driver that can only ask over a flaky `exec`:
# the first thing the job does is touch one file, the last is write its exit code
# to the other.
STARTED_FILE = "/tmp/run.started"
EXIT_CODE_FILE = "/tmp/exit_code"
# What `--ref` has to look like to be treated as a commit rather than a branch.
# `git.clone` takes the two on different keyword arguments and the server clones
# with `--branch`, which does not accept a SHA -- so a pinned commit passed as a
# branch fails provisioning in every sandbox rather than checking anything out.
COMMIT_RE = re.compile(r"[0-9a-f]{7,40}")
# `extractall(filter=...)` arrived in 3.11.4, so `requires-python = ">=3.11"` is
# not enough to assume it -- 3.11.2 raises TypeError. Feature-detect rather than
# version-compare, and ask for it where it exists: it is the default from 3.14,
# and until then an unfiltered extract trusts absolute paths inside the archive.
TAR_EXTRACT_KWARGS = (
    {"filter": "data"}
    if "filter" in inspect.signature(tarfile.TarFile.extractall).parameters
    else {}
)
# Keys worth forwarding. The harness exports every top-level key of
# credentials.toml as an environment variable, so an allowlist is what keeps an
# unrelated local key from being shipped to a third party by accident.
FORWARDED_KEYS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "API_VERSION",
    "OPENAI_API_KEY",
    "LMNR_PROJECT_API_KEY",
    "LMNR_BASE_URL",
    "MEDAGENTBENCH_FHIR_URL",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["biocoder", "biodsbench", "medagentbench", "medcalcbench"],
    )
    parser.add_argument(
        "--config-path", help="defaults to configs/gpt_5_6_luna/exp-gpt_5_6_luna-<task>.yaml"
    )
    parser.add_argument("--mode", default="test", help="train/test (default test)")
    parser.add_argument(
        "--indices-path", help="JSON of {task: {mode: [idx, ...]}} to run instead of a range"
    )
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1, help="-1 means all of data/metadata.json")
    parser.add_argument("--sandboxes", type=int, default=4, help="how many sandboxes to shard across")
    parser.add_argument("--n-jobs", type=int, default=1, help="joblib workers *inside* each sandbox")
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--result-dir-tag", help="overrides the config; also the Laminar session id")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument(
        "--ref",
        default="main",
        help="branch or tag to clone; anything that looks like a commit SHA is "
        "checked out detached instead",
    )
    parser.add_argument("--credentials", default=str(REPO_DIR / "credentials.toml"))
    parser.add_argument(
        "--snapshot",
        help="use this prebuilt Daytona snapshot instead of building Dockerfile.daytona",
    )
    parser.add_argument(
        "--output-dir", default=str(REPO_DIR), help="local directory the `workdir/` tree lands under"
    )
    parser.add_argument("--cpu", type=int, default=2, help="CPU cores per sandbox")
    parser.add_argument("--memory", type=int, default=4, help="GiB of RAM per sandbox")
    parser.add_argument("--disk", type=int, default=10, help="GiB of disk per sandbox")
    parser.add_argument("--timeout", type=int, default=4 * 60 * 60, help="seconds allowed per shard")
    parser.add_argument(
        "--with-fhir",
        action="store_true",
        help="give each shard its own HAPI FHIR sandbox (MedAgentBench only)",
    )
    parser.add_argument(
        "--keep-sandboxes", action="store_true", help="skip cleanup, to inspect a failure"
    )
    return parser.parse_args()


def load_credentials(path: str) -> dict[str, str]:
    """Read the local credentials file down to the keys the sandboxes need."""
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Copy credentials.example.toml and fill it in.")
    local = toml.load(path)
    creds = {k: str(v) for k, v in local.items() if k in FORWARDED_KEYS and v}
    if not creds:
        sys.exit(f"{path} has none of the keys the sandboxes need: {', '.join(FORWARDED_KEYS)}")
    return creds


def resolve_indices(args: argparse.Namespace) -> list[int]:
    """The full list of task indices to run, before sharding."""
    if args.indices_path:
        with open(args.indices_path) as f:
            return list(json.load(f)[args.task][args.mode])
    end_idx = args.end_idx
    if end_idx == -1:
        with open(REPO_DIR / "data" / "metadata.json") as f:
            end_idx = json.load(f)[args.task][args.mode]
    return list(range(args.start_idx, end_idx))


def shard(indices: list[int], n: int) -> list[list[int]]:
    """Deal the indices round-robin.

    Not contiguous slices: these files are grouped by source project and by
    calculator, so difficulty is correlated with position and contiguous shards
    would finish at wildly different times. A run costs as much wall clock as
    its slowest shard.
    """
    return [s for s in (indices[i::n] for i in range(n)) if s]


def run_remote(sandbox, command: str, timeout: int, label: str) -> str:
    """Run one command in a sandbox, and fail loudly with its output."""
    response = sandbox.process.exec(command, cwd=REMOTE_REPO, timeout=timeout)
    if response.exit_code != 0:
        raise RuntimeError(
            f"[{label}] exit code {response.exit_code} from:\n  {command}\n"
            f"{textwrap.indent(response.result or '', '  ')}"
        )
    return response.result or ""


def shard_state(sandbox) -> str:
    """Whether this sandbox's shard is `done`, `running`, `idle` or `unknown`.

    `unknown` means the sandbox would not answer, which is not the same as
    `idle`: the caller uses this to decide whether to start the job, and
    mistaking a running shard for an idle one starts a second copy of it.
    """
    # Marker files rather than `pgrep`: `procps` is not in the image, and a
    # missing `pgrep` reports every running shard as idle -- the one wrong answer
    # that does damage. The marker is the first thing the job writes.
    try:
        probe = sandbox.process.exec(
            f"if [ -f {EXIT_CODE_FILE} ]; then echo done; "
            f"elif [ -f {STARTED_FILE} ]; then echo running; "
            "else echo idle; fi",
            cwd=REMOTE_REPO,
            timeout=60,
        )
    except DaytonaError:
        return "unknown"
    return (probe.result or "").strip() or "unknown"


def launch(sandbox, inner: str, label: str, attempts: int = 5) -> None:
    """Start the shard in the background, retrying a flaky `exec`.

    Daytona's `exec` answers `command execution timeout` often enough under load
    to lose a shard on the one call that starts it -- and it says that whether or
    not the command it was given actually started, so a blind retry can leave two
    copies of the run racing over the same output files. Ask the sandbox first.
    """
    for attempt in range(1, attempts + 1):
        try:
            sandbox.process.exec(
                f"rm -f {EXIT_CODE_FILE} {STARTED_FILE} && "
                f"nohup bash -c {shlex.quote(inner)} > /tmp/run.log 2>&1 & echo started",
                cwd=REMOTE_REPO,
                timeout=60,
            )
            return
        except DaytonaError as e:
            print(f"[{label}] launch attempt {attempt}/{attempts} failed: {e}")
            time.sleep(15)
            state = shard_state(sandbox)
            if state in ("running", "done"):
                print(f"[{label}] the shard started anyway ({state})")
                return
            if state == "unknown":
                # Neither "it is running" nor "it is not", so starting another
                # one is not safe. Waiting is: the poll loop below tolerates a
                # silent sandbox, and `deadline` still bounds the whole thing.
                print(f"[{label}] sandbox not answering, waiting before retrying")
                time.sleep(45)
    raise RuntimeError(f"[{label}] could not start the shard in {attempts} attempts")


def run_detached(sandbox, argv: list[str], timeout: int, label: str) -> int:
    """Start `argv` in the background and poll until it writes its exit code.

    Not a plain `exec` with a long timeout: a shard runs for hours and holding
    one HTTP response open that long is at the mercy of every proxy in between.
    Polling also gives somewhere to report progress from, and leaves the run
    alive in the sandbox if this driver stumbles.

    Takes an argv rather than a string because the command is interpolated into
    a `bash -c` that is itself interpolated into an outer shell, so a config path
    or a `--result-dir-tag` holding a space or a quote would otherwise be split
    or terminate the quoting -- and the failure is silent, since the job that
    never starts also never writes the exit code this polls for.
    """
    inner = f"touch {STARTED_FILE}; {shlex.join(argv)}; echo $? > {EXIT_CODE_FILE}"
    launch(sandbox, inner, label)
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(30)
        # A probe that fails is a reason to poll again, not to give up on the
        # shard: the job runs detached and does not care whether anyone is
        # watching, while `exec` against a busy daemon returns "command
        # execution timeout" often enough to sink a multi-hour run over a `cat`.
        # `deadline` is what ends this loop.
        try:
            probe = sandbox.process.exec(
                f"cat {EXIT_CODE_FILE} 2>/dev/null", cwd=REMOTE_REPO, timeout=60
            )
            code = (probe.result or "").strip()
            if code:
                return int(code)
            written = sandbox.process.exec(
                "ls workdir/*/*/*/history_*.json 2>/dev/null | wc -l", cwd=REMOTE_REPO, timeout=60
            )
            print(f"[{label}] {(written.result or '0').strip()} trajectories written")
        except DaytonaError as e:
            print(f"[{label}] probe failed, still waiting: {e}")
    raise RuntimeError(f"[{label}] shard did not finish within {timeout}s")


def start_fhir_sandbox(client: Daytona, args: argparse.Namespace, label: str):
    """Bring up MedAgentBench's FHIR server as its own sandbox.

    This replaces upstream's `data/medagentbench/start_eval_docker.sh`: the
    `docker run` becomes a sandbox and the published port becomes a preview URL.

    One per shard, not one per run: a third of MedAgentBench's tasks POST to the
    server, and grading reads back what the agent wrote. Sharing a server across
    shards would let one shard's writes land in another shard's reads, so each
    shard starts from the image's pristine database and sees only itself.

    The sandbox is public because the workers have to reach it and carry no
    Daytona token; it holds only MedAgentBench's synthetic patients, but it is
    world-reachable for the life of the run.
    """
    label = f"{label}/fhir"
    print(f"[{label}] building sandbox image from Dockerfile.fhir ...")
    sandbox = client.create(
        CreateSandboxFromImageParams(
            image=Image.from_dockerfile(REPO_DIR / "Dockerfile.fhir"),
            public=True,
            os_user="root",
            labels={"medagentgym": "fhir", "shard": label},
            # HAPI is a JVM holding a ~1.4 GB H2 database open, so it wants more
            # RAM than a worker; 10 GB of disk is the per-sandbox ceiling on the
            # default Daytona plan and is enough for the image.
            resources=Resources(cpu=4, memory=8, disk=10),
            auto_stop_interval=0,
            ttl_minutes=args.timeout // 60 + 60,
        ),
        timeout=1800,
    )
    try:
        # Daytona replaces the image's entrypoint with its own agent, so the war
        # has to be launched by hand -- with the argv the original image had as
        # its entrypoint, since that is what wires the loader path to the war.
        sandbox.process.exec(
            "nohup java --class-path /app/main.war "
            "'-Dloader.path=main.war!/WEB-INF/classes/,main.war!/WEB-INF/,/app/extra-classes' "
            "org.springframework.boot.loader.PropertiesLauncher "
            "> /tmp/fhir.log 2>&1 & echo started",
            cwd="/app",
            timeout=60,
        )
        url = sandbox.get_preview_link(FHIR_PORT).url.rstrip("/") + "/fhir/"
        print(f"[{label}] waiting for {url} ...")
        # Polled from here rather than with a `curl` inside the sandbox: this is
        # the same public preview URL the workers will use, so a 200 here proves
        # the path that actually matters, and the JRE image has no curl anyway.
        for _ in range(90):
            time.sleep(10)
            try:
                with urllib.request.urlopen(f"{url}metadata", timeout=30) as resp:
                    if resp.status == 200:
                        print(f"[{label}] ready at {url}")
                        return sandbox, url
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
        tail = sandbox.process.exec("tail -40 /tmp/fhir.log", timeout=60)
        raise RuntimeError(
            f"FHIR server did not answer at {url} within 15 minutes.\n{tail.result}"
        )
    except BaseException:
        # The caller only learns about this sandbox if this function returns, so
        # anything that escapes has to take the sandbox with it or it leaks --
        # a stopped-but-billed container nobody has a handle on.
        if not args.keep_sandboxes:
            print(f"[{label}] startup failed, deleting sandbox")
            client.delete(sandbox)
        raise


def provision(client: Daytona, image, args: argparse.Namespace, creds: dict[str, str], label: str):
    """Create one sandbox and get the repo, its deps and the credentials into it."""
    common = {
        "os_user": "root",
        "labels": {"medagentgym": args.task, "shard": label},
        "resources": Resources(cpu=args.cpu, memory=args.memory, disk=args.disk),
        # A shard is hours of silence from Daytona's point of view, which the
        # default 15-minute idle timer would happily stop out from under us.
        # `ttl_minutes` is the backstop instead: wall clock from creation, so it
        # still fires if this driver dies before reaching its cleanup.
        "auto_stop_interval": 0,
        "ttl_minutes": args.timeout // 60 + 60,
    }
    if args.snapshot:
        params = CreateSandboxFromSnapshotParams(snapshot=args.snapshot, **common)
        sandbox = client.create(params, timeout=900)
    else:
        sandbox = client.create(
            CreateSandboxFromImageParams(image=image, **common),
            timeout=1800,
            on_snapshot_create_logs=lambda line: print(f"[image] {line.rstrip()}"),
        )
    try:
        return _fill_sandbox(sandbox, args, creds, label)
    except BaseException:
        # Everything past `create` is setup, and the caller only gets the handle
        # if this returns -- so a failure here leaks a running sandbox nobody
        # holds a reference to. One bad `git clone` used to leak the whole run's
        # worth of them at once.
        if not args.keep_sandboxes:
            print(f"[{label}] provisioning failed, deleting sandbox")
            client.delete(sandbox)
        raise


def _fill_sandbox(sandbox, args: argparse.Namespace, creds: dict[str, str], label: str):
    """Get the repo, its deps and the credentials into an already-created sandbox."""
    # Tags go on `branch=` -- `git clone --branch` takes either. A SHA cannot go
    # there, and `commit_id=` is not the answer either: the server resolves it
    # against the default branch, so a commit that lives on a feature branch
    # fails every sandbox with "object not found". Cloning the default branch and
    # fetching the commit by hand works wherever it lives. Hex is the only thing
    # distinguishing the two cases, so a branch named like a SHA would be
    # misread; `refs/heads/<name>` forces the branch reading if that ever bites.
    if COMMIT_RE.fullmatch(args.ref):
        sandbox.git.clone(args.repo_url, REMOTE_REPO)
        run_remote(
            sandbox,
            f"git fetch --depth 1 origin {shlex.quote(args.ref)} "
            "&& git checkout --detach FETCH_HEAD",
            timeout=600,
            label=label,
        )
    else:
        sandbox.git.clone(args.repo_url, REMOTE_REPO, branch=args.ref)
    # Nearly a no-op when the clone's lockfile matches the one baked into the
    # image, which is the point of baking it in; it still installs the project
    # itself, which the image deliberately skipped.
    run_remote(
        sandbox,
        f"UV_PROJECT_ENVIRONMENT={REMOTE_VENV} uv sync --locked --extra tasks",
        timeout=3600,
        label=label,
    )
    # Uploaded as a file rather than passed as `env_vars` so the secrets stay out
    # of the sandbox's metadata -- and because `set_environment_variables()`
    # exports every key of this file anyway, which is how MEDAGENTBENCH_FHIR_URL
    # reaches the task module.
    sandbox.fs.upload_file(toml.dumps(creds).encode(), f"{REMOTE_REPO}/credentials.toml")
    if args.task == "biodsbench":
        # ~160 MB of cBioPortal studies, gitignored, so the clone lacks them.
        run_remote(
            sandbox,
            f"{REMOTE_VENV}/bin/python scripts/fetch_biodsbench_data.py",
            timeout=1800,
            label=label,
        )
    return sandbox


def run_shard(client: Daytona, image, args: argparse.Namespace, creds: dict[str, str],
              indices: list[int], label: str) -> dict:
    """Provision a sandbox, run its shard, and bring the trajectories home."""
    sandbox = None
    fhir_sandbox = None
    try:
        if args.with_fhir:
            fhir_sandbox, fhir_url = start_fhir_sandbox(client, args, label)
            # Copied rather than mutated: the caller's dict is shared with every
            # other shard, each of which is pointed at a different server.
            creds = {**creds, "MEDAGENTBENCH_FHIR_URL": fhir_url}

        print(f"[{label}] provisioning for {len(indices)} tasks ...")
        sandbox = provision(client, image, args, creds, label)

        sandbox.fs.upload_file(
            json.dumps({args.task: {args.mode: indices}}).encode(), f"{REMOTE_REPO}/shard.json"
        )
        # One list element per argv word, so `run_detached` can quote each of
        # them individually on the way through two nested shells.
        argv = [
            f"{REMOTE_VENV}/bin/python",
            "main.py",
            "--config_path", args.config_path,
            "--rollout_indices_path", "shard.json",
            "--mode", args.mode,
            "--n_jobs", str(args.n_jobs),
            "--num_rollouts", str(args.num_rollouts),
        ]
        if args.num_steps:
            argv += ["--num_steps", str(args.num_steps)]
        if args.result_dir_tag:
            argv += ["--result_dir_tag", args.result_dir_tag]

        print(f"[{label}] running {len(indices)} tasks ...")
        exit_code = run_detached(sandbox, argv, args.timeout, label)
        if exit_code != 0:
            tail = run_remote(sandbox, "tail -40 /tmp/run.log", timeout=60, label=label)
            raise RuntimeError(f"main.py exited {exit_code}:\n{textwrap.indent(tail, '  ')}")

        # One tarball rather than a file at a time: a full run is thousands of
        # trajectories and each download would be its own round trip.
        # `running_records.jsonl` is excluded because it is one line per *run*,
        # so every shard would write its own partial success rate to the same
        # path and the last one extracted would win. The trajectories, which are
        # named per task index, do not collide.
        run_remote(
            sandbox,
            "tar czf /tmp/workdir.tgz --exclude=running_records.jsonl workdir",
            timeout=600,
            label=label,
        )
        blob = sandbox.fs.download_file("/tmp/workdir.tgz")
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            tar.extractall(args.output_dir, **TAR_EXTRACT_KWARGS)
        return {"label": label, "tasks": len(indices), "ok": True}
    except Exception as e:  # noqa: BLE001 - one bad shard must not sink the rest
        print(f"[{label}] FAILED: {e}")
        return {"label": label, "tasks": len(indices), "ok": False, "error": str(e)}
    finally:
        for doomed in (sandbox, fhir_sandbox):
            if doomed and not args.keep_sandboxes:
                try:
                    client.delete(doomed)
                except DaytonaError as e:
                    print(f"[{label}] could not delete sandbox: {e}")


def main() -> None:
    args = parse_args()
    if not args.config_path:
        args.config_path = f"configs/gpt_5_6_luna/exp-gpt_5_6_luna-{args.task}.yaml"
    if not os.getenv("DAYTONA_API_KEY"):
        sys.exit("DAYTONA_API_KEY is not set.")
    if args.with_fhir and args.task != "medagentbench":
        sys.exit("--with-fhir only makes sense for --task medagentbench.")

    creds = load_credentials(args.credentials)
    if args.task == "medagentbench" and not args.with_fhir and "MEDAGENTBENCH_FHIR_URL" not in creds:
        sys.exit(
            "medagentbench needs a FHIR server: pass --with-fhir, or put a reachable "
            "MEDAGENTBENCH_FHIR_URL in credentials.toml. Its default, localhost:8080, "
            "is not reachable from inside a sandbox."
        )

    indices = resolve_indices(args)
    shards = shard(indices, args.sandboxes)
    print(f"{len(indices)} tasks across {len(shards)} sandboxes ({args.n_jobs} joblib workers each)")

    client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    image = None if args.snapshot else Image.from_dockerfile(REPO_DIR / "Dockerfile.daytona")

    started = time.time()
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        results = list(
            pool.map(
                lambda pair: run_shard(client, image, args, creds, pair[1], f"shard-{pair[0]}"),
                enumerate(shards),
            )
        )

    ok = [r for r in results if r["ok"]]
    print("-" * 50)
    print(
        f"{len(ok)}/{len(results)} shards finished in {time.time() - started:.0f}s; "
        f"trajectories under {os.path.join(args.output_dir, 'workdir')}"
    )
    for failure in (r for r in results if not r["ok"]):
        print(f"  {failure['label']} failed: {failure['error']}")
    # A partially completed run is a failed run as far as a CI caller is concerned.
    sys.exit(0 if len(ok) == len(results) else 1)


if __name__ == "__main__":
    main()

"""Adopt an in-flight Daytona run whose driver is no longer around.

`daytona_run.py` starts each shard *detached* inside its sandbox, so the work
survives losing the driver: the job keeps running, and its trajectories keep
reaching Laminar, because the sandbox talks to the backend itself. What does not
survive is the tail of `run_shard` -- nothing downloads the `workdir/` tree and
nothing deletes the sandboxes, which then sit there until their TTL fires and
take the trajectory files with them.

This script is that tail, run separately: it finds the sandboxes a run left
behind, waits for each to write its exit code, downloads and unpacks its
`workdir/`, and deletes it. Sandboxes are matched by the `medagentgym` label
`daytona_run.py` sets, so nothing else on the account is touched.

    uv run --extra daytona python scripts/daytona_collect.py --task medcalcbench
    uv run --extra daytona python scripts/daytona_collect.py   # every task

Sandboxes carry a TTL measured from creation, and an adopted run has already
spent some of it, so `--extend-ttl` pushes the deadline out far enough to finish.
"""

import argparse
import io
import os
import sys
import tarfile
import time
from pathlib import Path

from daytona import Daytona, DaytonaConfig, DaytonaError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daytona_run import EXIT_CODE_FILE, REMOTE_REPO, TAR_EXTRACT_KWARGS  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent.parent
# The label value the FHIR sandboxes carry; they are servers for a shard rather
# than shards themselves, and are deleted with the shard they belong to.
FHIR_LABEL = "fhir"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", action="append", help="only these tasks (repeatable)")
    parser.add_argument(
        "--output-dir", default=str(REPO_DIR), help="local directory the `workdir/` tree lands under"
    )
    parser.add_argument("--interval", type=int, default=120, help="seconds between polls")
    parser.add_argument("--timeout", type=int, default=8 * 60 * 60, help="seconds to wait in total")
    parser.add_argument(
        "--extend-ttl",
        type=int,
        default=0,
        help="minutes to reset each sandbox's TTL to on adoption (0 leaves it alone)",
    )
    parser.add_argument(
        "--keep-sandboxes", action="store_true", help="download, but do not delete"
    )
    return parser.parse_args()


def probe(sandbox, cmd: str) -> str | None:
    """Run `cmd` in the sandbox, or return None if it could not be asked.

    A sandbox running a shard is busy, and `exec` against a busy daemon fails
    often enough that treating a failed probe as a verdict would throw away
    finished work. The caller polls again instead.
    """
    try:
        return (sandbox.process.exec(cmd, cwd=REMOTE_REPO, timeout=60).result or "").strip()
    except DaytonaError:
        return None


def collect(sandbox, args: argparse.Namespace, label: str) -> bool:
    """Download the shard's trajectories. False if the sandbox could not be read."""
    if probe(sandbox, "tar czf /tmp/workdir.tgz --exclude=running_records.jsonl workdir") is None:
        print(f"[{label}] finished, but the archive could not be built; retrying next poll")
        return False
    blob = sandbox.fs.download_file("/tmp/workdir.tgz")
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(args.output_dir, **TAR_EXTRACT_KWARGS)
    return True


def main() -> None:
    args = parse_args()
    if not os.getenv("DAYTONA_API_KEY"):
        sys.exit("DAYTONA_API_KEY is not set.")
    client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))

    everything = list(client.list())
    mine = [s for s in everything if s.labels.get("medagentgym")]
    workers = [s for s in mine if s.labels["medagentgym"] != FHIR_LABEL]
    if args.task:
        workers = [s for s in workers if s.labels["medagentgym"] in args.task]
    # Keyed by the shard label the worker will name when it is done with it.
    servers = {s.labels.get("shard"): s for s in mine if s.labels["medagentgym"] == FHIR_LABEL}
    if not workers:
        sys.exit("No sandboxes with a `medagentgym` label to collect.")
    print(f"adopting {len(workers)} shards ({len(servers)} FHIR servers) of {len(everything)} sandboxes")

    if args.extend_ttl:
        for sandbox in workers + list(servers.values()):
            try:
                sandbox.set_ttl(args.extend_ttl)
            except DaytonaError as e:
                print(f"could not extend TTL on {sandbox.id[:8]}: {e}")

    pending = {s.id: s for s in workers}
    failures = []
    deadline = time.time() + args.timeout
    while pending and time.time() < deadline:
        for sandbox_id, sandbox in list(pending.items()):
            label = f"{sandbox.labels['medagentgym']}/{sandbox.labels.get('shard')}"
            code = probe(sandbox, f"cat {EXIT_CODE_FILE} 2>/dev/null")
            if code is None:
                print(f"[{label}] unreachable, will ask again")
                continue
            if not code:
                done = probe(sandbox, "find workdir -name 'history_*.json' | wc -l")
                print(f"[{label}] running, {done or '?'} trajectories written")
                continue
            print(f"[{label}] exited {code}")
            if code != "0":
                failures.append((label, code, probe(sandbox, "tail -20 /tmp/run.log") or ""))
            if not collect(sandbox, args, label):
                continue
            del pending[sandbox_id]
            if args.keep_sandboxes:
                continue
            # The shard's FHIR server has no other client, so it goes with it.
            partner = servers.pop(f"{sandbox.labels.get('shard')}/{FHIR_LABEL}", None)
            for doomed in (sandbox, partner):
                if doomed:
                    try:
                        client.delete(doomed)
                    except DaytonaError as e:
                        print(f"[{label}] could not delete {doomed.id[:8]}: {e}")
        if pending:
            print(f"-- {len(pending)} shards still running, {deadline - time.time():.0f}s of budget left")
            time.sleep(args.interval)

    print("-" * 50)
    print(f"collected into {os.path.join(args.output_dir, 'workdir')}")
    for label, code, tail in failures:
        print(f"  {label} exited {code}:\n{tail}")
    if pending:
        for sandbox in pending.values():
            print(f"  still running at timeout: {sandbox.labels['medagentgym']}/{sandbox.labels.get('shard')}")
    sys.exit(1 if pending or failures else 0)


if __name__ == "__main__":
    main()

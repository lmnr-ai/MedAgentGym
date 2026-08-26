"""Drop BioCoder datapoints whose reference program is not gradable.

BioCoder has no stored ground truth: `BiocoderTask.setup()` derives it by
executing the reference program and capturing stdout. A datapoint is therefore
only usable if that program runs to completion and prints something stable. A
large minority of them do not, for three reasons:

* the `<<insert solution here>>` marker sits inside an expression
  (`num_matches = <<insert solution here>>`), so there is no way to splice a
  function *definition* in and get valid Python;
* the program imports something that is not installable (a module private to the
  project the snippet was scraped from, e.g. `your_module`, `my_mcmc_tools`);
* the program depends on network access, a missing input file, or the wall clock,
  so its output is not reproducible.

Upstream shipped these anyway. Their "ground truth" becomes a traceback that
embeds a per-run temporary filename, which no agent can ever reproduce, so they
silently score 0 for every model and only add cost.

Run this after changing the `tasks` extra -- the keep set is a function of what
is installed. Writes the filtered task files in place and updates
`data/metadata.json`.
"""

import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import failure_reason, run_python  # noqa: E402
from ehr_gym.env.task.substitution import insert_solution  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "biocoder"
METADATA = REPO / "data" / "metadata.json"


def check(item: tuple[int, dict]) -> dict:
    index, row = item
    code = insert_solution(row["context"], row["solution"])
    returncode, stdout, stderr = run_python(code)
    if returncode != 0:
        return {"index": index, "keep": False, "reason": failure_reason(stderr)}
    if not stdout.strip():
        # Ground truth would be the empty string, which any silent program matches.
        return {"index": index, "keep": False, "reason": "empty-output"}
    returncode, stdout_again, _ = run_python(code)
    if returncode != 0 or stdout_again != stdout:
        return {"index": index, "keep": False, "reason": "nondeterministic"}
    return {"index": index, "keep": True, "reason": ""}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true", help="report without rewriting")
    args = parser.parse_args()

    counts = {}
    for split in ("train", "test"):
        path = DATA / f"{split}_tasks.jsonl"
        rows = [json.loads(line) for line in path.open()]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(check, list(enumerate(rows)), chunksize=1))

        kept = [r["index"] for r in results if r["keep"]]
        dropped = collections.Counter(r["reason"] for r in results if not r["keep"])
        print(f"{split}: keeping {len(kept)} of {len(rows)}")
        for reason, count in dropped.most_common():
            print(f"  {count:5d}  {reason}")

        counts[split] = len(kept)
        if args.dry_run:
            continue
        with path.open("w") as out:
            for index in kept:
                # `code` was a pre-rendered copy of context-with-solution built by
                # the broken substitution; it is rebuilt at setup time now.
                row = {k: v for k, v in rows[index].items() if k != "code"}
                out.write(json.dumps(row) + "\n")

    if args.dry_run:
        return
    metadata = json.loads(METADATA.read_text())
    metadata["biocoder"] = counts
    METADATA.write_text(json.dumps(metadata, indent=4) + "\n")
    print(f"updated {METADATA} -> {counts}")


if __name__ == "__main__":
    main()

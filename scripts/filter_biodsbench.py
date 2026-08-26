"""Drop BioDSBench datapoints whose own reference solution fails their assertions.

A BioDSBench datapoint is graded by splicing the agent's code between the setup
code and a block of `assert` statements. That is only a fair test if the
reference solution shipped with the task passes those same assertions against the
data in `data/biodsbench/data/`. Run `scripts/fetch_biodsbench_data.py` first;
this script then rewrites the task files in place and updates
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
from ehr_gym.env.task.biodsbench import WORKDIR_PREFIX, _as_list  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "biodsbench"
METADATA = REPO / "data" / "metadata.json"

# A few references forget an import that the agent would obviously write itself
# (`os`, `pd`). Those tasks are still winnable, so the reference gets a second
# run with the usual preamble before we call the datapoint dead. `Agg` keeps the
# plotting tasks from blocking on a display we do not have.
PREAMBLE = """import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
"""


def check(item: tuple[int, dict]) -> dict:
    index, row = item
    study_dir = DATA / "data" / str(row["study_ids"])
    if not study_dir.is_dir():
        return {"index": index, "keep": False, "reason": f"no-data:{row['study_ids']}"}
    program = "\n".join(
        [
            "\n".join(_as_list(row["code_histories"])),
            row["reference_answer"],
            row["test_cases"],
        ]
    ).replace(WORKDIR_PREFIX, str(study_dir))
    returncode, _, stderr = run_python(program)
    if returncode == 0:
        return {"index": index, "keep": True, "reason": ""}
    returncode, _, _ = run_python(PREAMBLE + program)
    if returncode == 0:
        return {"index": index, "keep": True, "reason": ""}
    return {"index": index, "keep": False, "reason": failure_reason(stderr)}


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
        print(f"{split}: keeping {len(kept)} of {len(rows)}")
        for reason, count in collections.Counter(
            r["reason"] for r in results if not r["keep"]
        ).most_common():
            print(f"  {count:5d}  {reason}")

        counts[split] = len(kept)
        if args.dry_run:
            continue
        with path.open("w") as out:
            for index in kept:
                out.write(json.dumps(rows[index]) + "\n")

    if args.dry_run:
        return
    metadata = json.loads(METADATA.read_text())
    metadata["biodsbench"] = counts
    METADATA.write_text(json.dumps(metadata, indent=4) + "\n")
    print(f"updated {METADATA} -> {counts}")


if __name__ == "__main__":
    main()

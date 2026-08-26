"""Download and convert the cBioPortal study data that BioDSBench tasks read.

The task files in `data/biodsbench/` only describe the questions; the patient
data they operate on lives in the upstream HuggingFace dataset. This script
fetches the Python-task archive (~46 MB), keeps only the eleven studies our tasks
reference, and writes each file twice: cBioPortal's original tab-separated
`data_*.txt`, and the `data_*.csv` conversion most tasks expect. A handful of
tasks read the `.txt` directly, so both have to be there:

    data/biodsbench/data/<study_id>/data_clinical_patient.txt
                                   /data_clinical_patient.csv
                                   /data_mutations.txt
                                   ...

The result (~160 MB) is gitignored; run this once per checkout.
"""

import argparse
import csv
import io
import json
import re
import shutil
import tarfile
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASK_DIR = REPO / "data" / "biodsbench"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/zifeng-ai/BioDSBench/resolve/main/"
    "data_files/raw_patient_data_for_python_tasks.tar.gz"
)


def referenced_studies_and_files() -> tuple[set[str], set[str]]:
    """Read the task files to learn which studies and CSVs are actually used."""
    studies: set[str] = set()
    files: set[str] = set()
    for split in ("train", "test"):
        path = TASK_DIR / f"{split}_tasks.jsonl"
        for line in path.open():
            row = json.loads(line)
            studies.add(str(row["study_ids"]))
            files.update(re.findall(r"([\w-]+)\.(?:csv|txt)", json.dumps(row)))
    return studies, files


def to_csv(source: Path, destination: Path) -> int:
    """cBioPortal ships TSV with an optional block of `#` metadata lines on top."""
    with source.open(newline="", encoding="utf-8", errors="replace") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = list(csv.reader(lines, delimiter="\t"))
    with destination.open("w", newline="", encoding="utf-8") as out:
        csv.writer(out).writerows(rows)
    return max(len(rows) - 1, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="use an already-downloaded tarball instead of fetching"
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing data/ dir")
    args = parser.parse_args()

    destination_root = TASK_DIR / "data"
    if destination_root.exists():
        if not args.force:
            raise SystemExit(f"{destination_root} already exists; pass --force to replace it")
        shutil.rmtree(destination_root)

    studies, wanted = referenced_studies_and_files()
    print(f"{len(studies)} studies, {len(wanted)} distinct files referenced by the task files")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        archive = args.archive
        if archive is None:
            archive = tmpdir / "python_tasks.tar.gz"
            print(f"downloading {ARCHIVE_URL}")
            urllib.request.urlretrieve(ARCHIVE_URL, archive)
        print(f"extracting {archive}")
        with tarfile.open(archive) as tar:
            members = [
                m
                for m in tar.getmembers()
                if m.isfile()
                and Path(m.name).stem in wanted
                and Path(m.name).suffix == ".txt"
                and len(Path(m.name).parts) > 2
                and Path(m.name).parts[1] in studies
            ]
            tar.extractall(tmpdir, members=members)

        written = Counter()
        for extracted in sorted((tmpdir / "datasets").rglob("*.txt")):
            study = extracted.relative_to(tmpdir / "datasets").parts[0]
            study_dir = destination_root / study
            study_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(extracted, study_dir / extracted.name)
            rows = to_csv(extracted, study_dir / f"{extracted.stem}.csv")
            written[study] += 1
            print(f"  {study}/{extracted.stem}.{{txt,csv}}  ({rows} rows)")

    missing = studies - set(written)
    if missing:
        raise SystemExit(f"archive did not contain data for studies: {sorted(missing)}")
    total = sum(f.stat().st_size for f in destination_root.rglob("*") if f.is_file())
    print(f"wrote {sum(written.values())} files across {len(written)} studies "
          f"({total / 1024 ** 2:.0f} MB) to {destination_root}")


if __name__ == "__main__":
    main()

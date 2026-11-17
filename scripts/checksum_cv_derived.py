#!/usr/bin/env python3
"""Add or verify sha256 checksums for the frozen Common Voice derived audio set.

The set built by scripts/build_cv_distorted.py is frozen: Praat's
"Change gender" is not bit-reproducible run to run, so regenerating it would
give slightly different audio even with the same seeds. These checksums pin
the exact files that were generated.

  add    (default) write clean_sha256 to clean_16k_index.csv and
         distorted_sha256 to distortion_params.csv; refuses to overwrite
         existing checksums
  verify recompute every checksum and exit 1 on any missing or changed file

Paths in the CSVs are relative to the data root (--out-dir).
"""

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import cv_data_root, resolve_data_path  # noqa: E402

TABLES = (
    ("clean_16k_index.csv", "clean_path", "clean_sha256"),
    ("distortion_params.csv", "distorted_path", "distorted_sha256"),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_table(path, columns, rows):
    tmp = path.with_suffix(".tmp.csv")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def add(out_dir):
    for name, path_col, sha_col in TABLES:
        path = out_dir / name
        columns, rows = read_table(path)
        if sha_col in columns:
            sys.exit(f"{path} already has {sha_col}; the set is frozen, use 'verify'")
        hashed = 0
        for row in rows:
            row[sha_col] = sha256_file(resolve_data_path(row[path_col], out_dir)) if row[path_col] else ""
            hashed += bool(row[sha_col])
        write_table(path, columns + [sha_col], rows)
        print(f"{name}: {hashed} file checksum(s) added ({len(rows)} rows)")


REPO_MANIFESTS = Path(__file__).resolve().parents[1] / "data" / "manifests"


def verify(out_dir, from_repo=False):
    problems = 0
    for name, path_col, sha_col in TABLES:
        table = REPO_MANIFESTS / f"cv27_th_{name}" if from_repo else out_dir / name
        print(f"reading {table}")
        columns, rows = read_table(table)
        if sha_col not in columns:
            sys.exit(f"{name} has no {sha_col} column; run 'add' first")
        checked = 0
        for row in rows:
            if not row[path_col]:
                continue
            checked += 1
            path = resolve_data_path(row[path_col], out_dir)
            if not path.is_file():
                print(f"MISSING  {row[path_col]}")
                problems += 1
            elif sha256_file(path) != row[sha_col]:
                print(f"CHANGED  {row[path_col]}")
                problems += 1
        print(f"{name}: {checked} file(s) checked")
    print("OK" if not problems else f"{problems} problem(s)")
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=("add", "verify"), default="add")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Data root holding the CSVs and audio (default: $DATA_ROOT or "
                             "configs/data.yaml cv27_data_root)")
    parser.add_argument("--from-repo", action="store_true",
                        help="verify: read the checksums from the git-tracked copies in "
                             "data/manifests/ instead of the CSVs in the data root")
    args = parser.parse_args()
    out_dir = (args.out_dir or cv_data_root()).expanduser()
    if args.mode == "add":
        add(out_dir)
        return 0
    return verify(out_dir, args.from_repo)


if __name__ == "__main__":
    sys.exit(main())

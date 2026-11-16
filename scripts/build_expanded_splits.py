#!/usr/bin/env python3
"""Extend the W5 utterance split with new utterances without moving old ones.

Every utterance already assigned in the W5 split keeps its assignment, so the
W5 validation utterances (and the 8 used for STOI/PESQ) stay in validation.
New utterances are split with the same ratios and seed as W5.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.splits import SPLIT_NAMES, create_utterance_splits  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-splits", default="data/splits_w5.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--severity", default="severe")
    args = parser.parse_args()

    base = json.loads(Path(args.base_splits).read_text(encoding="utf-8"))
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["severity"] == args.severity]
    all_ids = set(manifest["utterance_id"].astype(str))

    base_ids = {uid for name in SPLIT_NAMES for uid in base["splits"][name]}
    missing = sorted(base_ids - all_ids)
    if missing:
        print(f"warning: {len(missing)} W5 utterance(s) missing from manifest: {missing}")
    new_ids = sorted(all_ids - base_ids)
    new_splits = create_utterance_splits(new_ids, base["ratios"], base["seed"])

    splits = {
        name: sorted((set(base["splits"][name]) & all_ids) | set(new_splits[name]))
        for name in SPLIT_NAMES
    }
    assigned = [uid for name in SPLIT_NAMES for uid in splits[name]]
    assert len(assigned) == len(set(assigned)) and set(assigned) == all_ids

    payload = {
        "seed": base["seed"],
        "ratios": base["ratios"],
        "splits": splits,
        "base_splits": args.base_splits,
        "new_utterance_splits": {name: len(new_splits[name]) for name in SPLIT_NAMES},
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")
    for name in SPLIT_NAMES:
        print(
            f"  {name}: {len(splits[name])} utterance(s) "
            f"(W5 {len(set(base['splits'][name]) & all_ids)} + new {len(new_splits[name])})"
        )


if __name__ == "__main__":
    main()

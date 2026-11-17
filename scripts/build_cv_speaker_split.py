#!/usr/bin/env python3
"""Build a speaker-independent clip selection from a Common Voice Thai release.

Metadata only: reads ``validated.tsv`` and ``clip_durations.tsv`` from an
extracted Common Voice release and writes one CSV row per selected clip with
its speaker-level split. No audio is read, converted or copied.

Selection
  1. validated clips with up_votes >= 2 and down_votes == 0
  2. duration within [--min-sec, --max-sec]
  3. per client_id, at most one clip per distinct sentence
  4. speakers with >= --min-per-speaker clips; at most --max-per-speaker kept
  5. speakers are taken in a seeded order until about --target-clips clips
  6. sentence-disjoint splits: val, then test, skip sentences used by another
     split; a speaker left with too few clips is replaced by the next unused
     speaker of the same gender

Split
  Speakers (not clips) are assigned train/val/test by --ratios, stratified by
  self-declared gender, so every client_id belongs to exactly one split.

Every ordering uses sha256("<seed>:<key>") rather than a shared RNG, so the
result does not depend on the row order of the .tsv files.
"""

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")
COLUMNS = ["clip_id", "client_id", "split", "sentence", "duration", "gender", "age"]
DEFAULT_OUTPUT = Path("data/manifests/cv27_th_speaker_split.csv")


def stable_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def read_tsv(path):
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                       keep_default_na=False)


def speaker_label(values, missing):
    """Most frequent non-empty self-declared value for a speaker (ties: sorted)."""
    counts = Counter(v for v in values if v)
    if not counts:
        return missing
    best = max(counts.values())
    return sorted(v for v, n in counts.items() if n == best)[0]


def load_candidates(cv_dir, min_sec, max_sec):
    cv_dir = Path(cv_dir)
    validated = read_tsv(cv_dir / "validated.tsv")
    durations = read_tsv(cv_dir / "clip_durations.tsv")
    durations["duration"] = durations["duration[ms]"].astype(int) / 1000.0
    clips = validated.merge(durations[["clip", "duration"]], left_on="path",
                            right_on="clip", how="left", validate="one_to_one")
    if clips["duration"].isna().any():
        raise ValueError(f"{clips['duration'].isna().sum()} clips have no duration")

    clips = clips[(clips["up_votes"].astype(int) >= 2)
                  & (clips["down_votes"].astype(int) == 0)
                  & (clips["client_id"] != "")
                  & (clips["duration"] >= min_sec)
                  & (clips["duration"] <= max_sec)].copy()
    clips["clip_id"] = clips["path"].str.replace(r"\.mp3$", "", regex=True)
    return clips


def select_speakers(clips, seed, min_per_speaker, max_per_speaker, target_clips):
    """Order eligible speakers by seed; the first ones reaching ~target_clips are chosen.

    Returns (chosen, reserve, candidates, n_eligible). ``reserve`` lists the
    remaining eligible speakers in the same seeded order, used as replacements.
    ``candidates`` holds each eligible speaker's clips, one per sentence, in
    seeded order.
    """
    clips = clips.copy()
    clips["_order"] = [stable_key(seed, c) for c in clips["clip_id"]]
    clips = clips.sort_values(["client_id", "_order"])
    clips = clips.drop_duplicates(["client_id", "sentence"], keep="first")

    per_speaker = clips.groupby("client_id").size()
    eligible = per_speaker[per_speaker >= min_per_speaker].index
    speakers = sorted(eligible, key=lambda s: stable_key(seed, s))

    chosen, total = [], 0
    for speaker in speakers:
        if total >= target_clips:
            break
        chosen.append(speaker)
        total += min(int(per_speaker[speaker]), max_per_speaker)

    reserve = speakers[len(chosen):]
    candidates = clips[clips["client_id"].isin(set(eligible))].drop(columns="_order")
    return chosen, reserve, candidates, len(eligible)


def speaker_meta(candidates):
    return candidates.groupby("client_id").agg(
        gender=("gender", lambda s: speaker_label(s, "unspecified")),
        age=("age", lambda s: speaker_label(s, "unspecified")),
    )


def split_counts(n, ratios):
    raw = [n * r for r in ratios]
    counts = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True):
        if sum(counts) == n:
            break
        counts[i] += 1
    return counts


def assign_speaker_splits(meta, seed, ratios):
    """Split speakers train/val/test by ``ratios`` within each gender group."""
    assignment = {}
    for _, group in meta.groupby("gender"):
        speakers = sorted(group.index, key=lambda s: stable_key(f"{seed}:split", s))
        start = 0
        for name, count in zip(SPLITS, split_counts(len(speakers), ratios)):
            for speaker in speakers[start:start + count]:
                assignment[speaker] = name
            start += count
    return assignment


def pick_clips(candidates, assignment, chosen, reserve, meta,
               min_per_speaker, max_per_speaker):
    """Pick clips split by split so that no sentence is shared between splits.

    Train speakers take their first clips; val and then test speakers skip any
    sentence already used by another split. A val/test speaker left with fewer
    than ``min_per_speaker`` clips is dropped and replaced by the next reserve
    speaker of the same gender, so split sizes and gender balance are kept.
    """
    by_speaker = {s: g for s, g in candidates.groupby("client_id", sort=False)}
    reserve_by_gender = {}
    for speaker in reserve:
        reserve_by_gender.setdefault(meta.at[speaker, "gender"], []).append(speaker)

    initial = {name: [s for s in chosen if assignment[s] == name] for name in SPLITS}
    assignment = dict(assignment)
    used = {name: set() for name in SPLITS}
    picked = []
    report = {name: {"dropped": [], "replacements": []} for name in SPLITS}
    for name in SPLITS:
        blocked = set().union(*(used[other] for other in SPLITS if other != name))
        for speaker in initial[name]:
            queue = reserve_by_gender.get(meta.at[speaker, "gender"], [])
            current = speaker
            while current is not None:
                group = by_speaker[current]
                group = group[~group["sentence"].isin(blocked)].head(max_per_speaker)
                if len(group) >= min_per_speaker:
                    assignment[current] = name
                    used[name].update(group["sentence"])
                    picked.append(group)
                    if current != speaker:
                        report[name]["replacements"].append(current)
                    break
                report[name]["dropped"].append(current)
                assignment.pop(current, None)
                current = queue.pop(0) if queue else None
    return pd.concat(picked), assignment, report


def build(cv_dir, seed, min_sec, max_sec, min_per_speaker, max_per_speaker,
          target_clips, ratios):
    clips = load_candidates(cv_dir, min_sec, max_sec)
    chosen, reserve, candidates, n_eligible = select_speakers(
        clips, seed, min_per_speaker, max_per_speaker, target_clips)
    meta = speaker_meta(candidates)
    assignment = assign_speaker_splits(meta.loc[chosen], seed, ratios)
    selected, assignment, report = pick_clips(
        candidates, assignment, chosen, reserve, meta, min_per_speaker, max_per_speaker)
    out = pd.DataFrame({
        "clip_id": selected["clip_id"],
        "client_id": selected["client_id"],
        "split": selected["client_id"].map(assignment),
        "sentence": selected["sentence"],
        "duration": selected["duration"].round(3),
        "gender": selected["client_id"].map(meta["gender"]),
        "age": selected["client_id"].map(meta["age"]),
    })
    out["_s"] = out["split"].map({name: i for i, name in enumerate(SPLITS)})
    out = out.sort_values(["_s", "client_id", "clip_id"]).drop(columns="_s")
    stats = {"candidates": len(clips), "eligible_speakers": n_eligible,
             "chosen_speakers": len(chosen), "report": report}
    return out.reset_index(drop=True), stats


def print_report(df, stats, args):
    print(f"candidate clips (votes + {args.min_sec}-{args.max_sec}s): {stats['candidates']}")
    print(f"speakers with >= {args.min_per_speaker} usable clips: {stats['eligible_speakers']}")
    print(f"chosen speakers: {stats['chosen_speakers']}")
    for name, item in stats["report"].items():
        print(f"  {name}: {len(item['dropped'])} speaker(s) dropped (< {args.min_per_speaker} "
              f"clips without cross-split sentences), {len(item['replacements'])} replaced from reserve")
    print(f"selected: {df['client_id'].nunique()} speakers, {len(df)} clips, "
          f"{df['duration'].sum() / 3600:.2f} h\n")
    print(f"{'split':6s} {'speakers':>8s} {'clips':>6s} {'hours':>6s}  gender (speakers)")
    for name in SPLITS:
        part = df[df["split"] == name]
        genders = part.drop_duplicates("client_id")["gender"].value_counts().to_dict()
        print(f"{name:6s} {part['client_id'].nunique():8d} {len(part):6d} "
              f"{part['duration'].sum() / 3600:6.2f}  {genders}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cv-dir", required=True,
                        help="Extracted release locale folder, e.g. .../cv-corpus-27.0-2026-09-11/th")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sec", type=float, default=2.5)
    parser.add_argument("--max-sec", type=float, default=8.0)
    parser.add_argument("--min-per-speaker", type=int, default=4)
    parser.add_argument("--max-per-speaker", type=int, default=8)
    parser.add_argument("--target-clips", type=int, default=3000)
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()
    if abs(sum(args.ratios) - 1.0) > 1e-9:
        parser.error("--ratios must sum to 1")

    df, stats = build(args.cv_dir, args.seed, args.min_sec, args.max_sec,
                      args.min_per_speaker, args.max_per_speaker,
                      args.target_clips, args.ratios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, columns=COLUMNS, encoding="utf-8")
    print_report(df, stats, args)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

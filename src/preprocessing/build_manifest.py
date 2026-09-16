"""Build a manifest from one or more clean/distorted directory pairs."""

import argparse
import csv
import os
from pathlib import Path

import pandas as pd
import soundfile as sf


DEFAULT_CLEAN_DIRS = [os.path.join("data", "raw")]
DEFAULT_DISTORTED_DIRS = [os.path.join("data", "distorted")]
DEFAULT_MANIFEST_PATH = os.path.join("data", "manifest.csv")
SEVERITY_LEVELS = ("mild", "moderate", "severe")

COLUMNS = [
    "utterance_id",
    "speaker_id",
    "clean_path",
    "distorted_path",
    "severity",
    "transcript",
    "duration_sec",
]


def extract_transcript(utterance_id):
    """Recover a transcript from legacy files named ``001_transcript.wav``."""
    if "_" not in utterance_id:
        return ""
    return utterance_id.split("_", 1)[1]


def load_transcript_metadata(clean_dir):
    """Load optional ``transcripts.csv`` metadata indexed by WAV stem."""
    transcript_path = Path(clean_dir) / "transcripts.csv"
    if not transcript_path.is_file():
        return {}

    with transcript_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "sentence"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{transcript_path} must contain columns: filename, sentence"
            )
        metadata = {}
        for row in reader:
            utterance_id = Path(row["filename"]).stem
            if utterance_id in metadata:
                raise ValueError(
                    f"Duplicate filename in {transcript_path}: {row['filename']}"
                )
            metadata[utterance_id] = row
    return metadata


def default_speaker_id(clean_dir):
    """Preserve the original speaker ID and label other sources by directory."""
    source_name = Path(clean_dir).name
    return "spk001" if source_name == "raw" else source_name


def _as_directory_list(value):
    if isinstance(value, (str, os.PathLike)):
        return [os.fspath(value)]
    return [os.fspath(path) for path in value]


def validate_directory_pairs(clean_dirs, distorted_dirs):
    if len(clean_dirs) != len(distorted_dirs):
        raise ValueError(
            "--clean_dirs and --distorted_dirs must contain the same number "
            f"of directories (got {len(clean_dirs)} and {len(distorted_dirs)})"
        )
    for directory in [*clean_dirs, *distorted_dirs]:
        if not Path(directory).is_dir():
            raise FileNotFoundError(f"Directory does not exist: {directory}")


def build_manifest_rows(clean_dirs, distorted_dirs, severities=SEVERITY_LEVELS):
    """Build rows from corresponding clean/distorted directory pairs."""
    clean_dirs = _as_directory_list(clean_dirs)
    distorted_dirs = _as_directory_list(distorted_dirs)
    validate_directory_pairs(clean_dirs, distorted_dirs)

    rows = []
    seen_utterances = set()
    for clean_dir, distorted_dir in zip(clean_dirs, distorted_dirs):
        metadata = load_transcript_metadata(clean_dir)
        source_speaker_id = default_speaker_id(clean_dir)

        for clean_path in sorted(Path(clean_dir).glob("*.wav")):
            utterance_id = clean_path.stem
            if utterance_id in seen_utterances:
                raise ValueError(
                    f"Duplicate utterance_id across clean directories: {utterance_id}"
                )
            seen_utterances.add(utterance_id)

            metadata_row = metadata.get(utterance_id, {})
            transcript = metadata_row.get("sentence") or extract_transcript(utterance_id)
            speaker_id = metadata_row.get("speaker_id") or source_speaker_id

            for severity in severities:
                distorted_path = Path(distorted_dir) / f"{utterance_id}_{severity}.wav"
                if not distorted_path.is_file():
                    print(f"  warning: missing distorted file, skipping: {distorted_path}")
                    continue

                rows.append({
                    "utterance_id": utterance_id,
                    "speaker_id": speaker_id,
                    "clean_path": os.fspath(clean_path),
                    "distorted_path": os.fspath(distorted_path),
                    "severity": severity,
                    "transcript": transcript,
                    "duration_sec": round(sf.info(distorted_path).duration, 4),
                })
    return rows


def build_manifest_df(clean_dirs, distorted_dirs, severities=SEVERITY_LEVELS):
    rows = build_manifest_rows(clean_dirs, distorted_dirs, severities)
    dataframe = pd.DataFrame(rows, columns=COLUMNS)
    return dataframe.sort_values(["utterance_id", "severity"]).reset_index(drop=True)


def print_summary(dataframe, output_path, severities):
    print(
        f"\nWrote {len(dataframe)} row(s) covering "
        f"{dataframe['utterance_id'].nunique()} utterance(s) to {output_path}"
    )
    counts = dataframe["severity"].value_counts().reindex(severities, fill_value=0)
    for severity, count in counts.items():
        print(f"  {severity}: {count} rows")
    print(f"  total distorted audio duration: {dataframe['duration_sec'].sum():.1f} sec")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean_dirs", nargs="+", default=DEFAULT_CLEAN_DIRS,
        help="Clean-audio directories; order must match --distorted_dirs",
    )
    parser.add_argument(
        "--distorted_dirs", nargs="+", default=DEFAULT_DISTORTED_DIRS,
        help="Distorted-audio directories; order must match --clean_dirs",
    )
    parser.add_argument(
        "--severities", nargs="+", choices=SEVERITY_LEVELS,
        default=list(SEVERITY_LEVELS),
    )
    parser.add_argument("--output", default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    print("Scanning directory pairs:")
    for clean_dir, distorted_dir in zip(args.clean_dirs, args.distorted_dirs):
        print(f"  clean={clean_dir} | distorted={distorted_dir}")

    dataframe = build_manifest_df(
        args.clean_dirs,
        args.distorted_dirs,
        severities=args.severities,
    )
    if dataframe.empty:
        raise RuntimeError("No manifest rows generated; check the input directories")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8")
    print_summary(dataframe, output_path, args.severities)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download and prepare validated Mozilla Common Voice Thai clips."""

import argparse
import csv
import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from datasets.data_files import EmptyDatasetError


DATASET_NAME = "mozilla-foundation/common_voice_17_0"
FALLBACK_DATASET_NAME = "fsicoli/common_voice_17_0"
DATASET_CONFIG = "th"
TARGET_SAMPLE_RATE = 16000
DEFAULT_LIMIT = 200
DEFAULT_OUTPUT_DIR = Path("data/common_voice")


def decode_audio(audio_record):
    """Decode an Audio(decode=False) record into mono float32 samples."""
    if audio_record is None:
        return None
    audio_bytes = audio_record.get("bytes")
    audio_path = audio_record.get("path")
    if audio_bytes is not None:
        source = io.BytesIO(audio_bytes)
    elif audio_path:
        source = audio_path
    else:
        return None

    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
        ).astype(np.float32)
    return audio


def prepare(output_dir=DEFAULT_OUTPUT_DIR, limit=DEFAULT_LIMIT):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Move or remove it before rerunning."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split="train",
            trust_remote_code=True,
        )
        source_name = DATASET_NAME
    except EmptyDatasetError:
        print(
            f"แหล่งทางการ {DATASET_NAME} ไม่มี data files แล้ว; "
            f"ใช้ mirror {FALLBACK_DATASET_NAME}"
        )
        dataset = load_dataset(
            FALLBACK_DATASET_NAME,
            DATASET_CONFIG,
            split="train",
            trust_remote_code=True,
        )
        source_name = FALLBACK_DATASET_NAME
    dataset = dataset.cast_column("audio", Audio(decode=False))
    validated = dataset.filter(
        lambda up_votes, down_votes: up_votes >= 2 and down_votes == 0,
        input_columns=["up_votes", "down_votes"],
        desc="Filtering validated clips",
    )

    transcript_rows = []
    total_duration = 0.0
    decode_failures = 0
    for row in validated:
        if row["audio"] is None:
            continue
        try:
            audio = decode_audio(row["audio"])
        except Exception as exc:
            decode_failures += 1
            print(f"คำเตือน: ข้ามไฟล์ที่ถอดรหัสไม่ได้ ({exc})")
            continue
        if audio is None or audio.size == 0:
            continue

        index = len(transcript_rows)
        filename = f"cv_{index:04d}.wav"
        sf.write(
            output_dir / filename,
            audio,
            TARGET_SAMPLE_RATE,
            subtype="PCM_16",
        )
        transcript_rows.append({"filename": filename, "sentence": row["sentence"]})
        total_duration += len(audio) / TARGET_SAMPLE_RATE
        if len(transcript_rows) >= limit:
            break

    if len(transcript_rows) < limit:
        raise RuntimeError(
            f"Found only {len(transcript_rows)} usable clips; requested {limit}. "
            f"Decode failures: {decode_failures}"
        )

    transcript_path = output_dir / "transcripts.csv"
    with transcript_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "sentence"])
        writer.writeheader()
        writer.writerows(transcript_rows)

    print("\nสรุป Common Voice Thai")
    print(f"แหล่งข้อมูล: {source_name}")
    print(f"ไฟล์ที่บันทึก: {len(transcript_rows)}")
    print(f"ระยะเวลารวม: {total_duration:.2f} วินาที")
    print(f"อัตราสุ่มตัวอย่าง: {TARGET_SAMPLE_RATE} Hz, mono")
    print("ตัวอย่าง transcript 5 รายการแรก:")
    for row in transcript_rows[:5]:
        print(f"  {row['filename']}: {row['sentence']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    prepare(output_dir=args.output_dir, limit=args.limit)


if __name__ == "__main__":
    main()

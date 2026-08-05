#!/usr/bin/env python3
"""Record speech utterances from the microphone into data/raw/ (or --output_dir)."""
import argparse
import os
import re
import time

import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION_SECONDS = 5
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
)


def sanitize_label(label: str) -> str:
    label = label.strip()
    label = re.sub(r'[\\/:*?"<>|]', "", label)
    label = re.sub(r"\s+", "_", label)
    return label


def next_index(output_dir: str) -> int:
    existing = [f for f in os.listdir(output_dir) if re.match(r"^\d{3}_", f)]
    if not existing:
        return 1
    return max(int(f[:3]) for f in existing) + 1


def countdown():
    for n in (3, 2, 1):
        print(n)
        time.sleep(1)
    print("เริ่มบันทึก!")


def record_utterance():
    audio = sd.rec(
        int(DURATION_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    return audio


def main():
    parser = argparse.ArgumentParser(description="Record speech utterances from the microphone.")
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save recorded .wav files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    print("=== เครื่องมือบันทึกเสียง ===")
    print(f"บันทึกไปที่: {output_dir}")
    print(f"อัตราสุ่มตัวอย่าง: {SAMPLE_RATE} Hz | ความยาว: {DURATION_SECONDS} วินาทีต่อประโยค\n")

    while True:
        label = input("พิมพ์ประโยค/ป้ายกำกับที่ต้องการอ่าน แล้วกด Enter: ").strip()
        if not label:
            print("กรุณาพิมพ์ประโยคก่อนบันทึก\n")
            continue

        input("กด Enter เมื่อพร้อมเริ่มบันทึก...")
        countdown()

        audio = record_utterance()

        index = next_index(output_dir)
        filename = f"{index:03d}_{sanitize_label(label)}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, audio, SAMPLE_RATE, subtype="PCM_16")

        print(f"บันทึกสำเร็จ: {filepath}\n")

        again = input("ต้องการบันทึกประโยคถัดไปหรือไม่? (y/n): ").strip().lower()
        if again not in ("y", "yes", "ใช่"):
            print("จบการบันทึกเสียง")
            break


if __name__ == "__main__":
    main()

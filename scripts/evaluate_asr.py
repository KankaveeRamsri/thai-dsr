#!/usr/bin/env python3
"""ASR intelligibility (CER primary, WER secondary) on the CV 27.0 test split.

Each system's OWN output is transcribed: its native-timeline audio, with no
inverse warp, since ASR does not need time alignment. The transcript is scored
against the reference sentence (the Common Voice prompt).

  clean                 the untouched clean clip; one transcript per clip,
                        repeated for the three severities so the table lines up
  distorted             the distorted input (floor)
  <evaluated system>    <eval-dir>/<system>/audio/<clip>_<severity>.wav as written
                        by evaluate_cv_aligned.py, e.g. copysynth_tempo,
                        mapper_tempo, mapper_linear_deploy

Model, revision and decoding are pinned in src/evaluation/asr.py. Text
normalisation, CER and WER are defined in src/evaluation/thai_text.py.

Resumable: every transcript is appended to <eval-dir>/<system>/asr_transcripts.jsonl
(one JSON line per clip, flushed and fsynced) as soon as its batch is decoded,
together with a fingerprint of the ASR and text settings. On restart, clips
that already have a transcript with the same fingerprint are skipped, so an
interrupted run loses at most one batch. A transcript with a different
fingerprint stops the run; delete the file or use --overwrite. CER / WER are
always recomputed from the stored transcripts.

Writes <eval-dir>/<system>/asr_metrics.csv (one row per clip x severity:
hypothesis, normalised texts, cer, wer, edit counts, reference lengths) and
asr_run_info.json. For clean it also writes a minimal per_utterance.csv, so
scripts/summarize_cv_eval.py can list it as a system. Corpus-level CER / WER
(sum of edits / sum of reference lengths) and the speaker-level bootstrap are
computed by the summarizer.

--limit N scores N random test clips (seed 42) per system: the sanity check.
Write those runs to a separate --eval-dir.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.asr import ThaiASR  # noqa: E402
from src.evaluation.metrics import load_audio_16k  # noqa: E402
from src.evaluation.thai_text import error_rates, text_settings  # noqa: E402
from src.utils.config import cv_data_root  # noqa: E402

DEFAULT_SYSTEMS = ["clean", "distorted", "copysynth_tempo", "mapper_tempo", "mapper_linear_deploy"]
SEVERITIES = ["mild", "moderate", "severe"]


def git_state():
    def run(*cmd):
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def fingerprint(asr):
    settings = {"asr": {k: v for k, v in asr.settings().items()}, "text": text_settings()}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16], settings


def load_transcripts(path, expected):
    """(clip, severity) -> stored record, for records made with the current fingerprint."""
    done = {}
    if not path.exists():
        return done
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):  # killed mid-line: start the next record on a fresh line
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n")
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line cut off by a kill: ignore it, it is redone
            print(f"  {path.name}:{line_no}: incomplete line ignored", flush=True)
            continue
        if record["fingerprint"] != expected:
            raise SystemExit(f"{path}:{line_no} was made with other ASR/text settings "
                             f"({record['fingerprint']} != {expected}); delete it or use --overwrite")
        done[(record["utterance_id"], record["severity"])] = record
    return done


def get_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=None, help="default: <data root>/training_manifest.csv")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/manifests/cv27_th_splits.json")
    parser.add_argument("--speakers", type=Path, default=ROOT / "data/manifests/cv27_th_speaker_split.csv")
    parser.add_argument("--audio-dir", type=Path, default=None,
                        help="where <system>/audio/ lives (default: <data root>/eval/test_full)")
    parser.add_argument("--eval-dir", type=Path, default=None, help="output (default: <data root>/eval/test_full)")
    parser.add_argument("--limit", type=int, default=0, help="N random test clips (seed 42): sanity check")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="pinned: cpu (see src/evaluation/asr.py)")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32",
                        help="pinned: float32 (see src/evaluation/asr.py)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = cv_data_root()
    audio_dir = args.audio_dir or root / "eval" / "test_full"
    eval_dir = args.eval_dir or root / "eval" / "test_full"
    manifest = pd.read_csv(args.manifest or root / "training_manifest.csv")
    clips = sorted(json.loads(args.splits.read_text(encoding="utf-8"))["splits"][args.split])
    if args.limit:
        clips = sorted(random.Random(42).sample(clips, args.limit))
    rows = manifest[manifest["utterance_id"].isin(set(clips))].sort_values(["severity", "utterance_id"])
    rows = rows[rows["severity"].isin(SEVERITIES)].reset_index(drop=True)
    assert len(rows) == 3 * len(clips), (len(rows), len(clips))
    speaker_of = dict(pd.read_csv(args.speakers, usecols=["clip_id", "client_id"]).values)
    reference = dict(zip(rows["utterance_id"], rows["transcript"]))

    device = get_device(args.device)
    asr = ThaiASR(device, dtype=getattr(torch, args.dtype))
    fp, fp_settings = fingerprint(asr)
    print(f"ASR {asr.model_id}@{asr.revision[:12]} on {device} ({args.dtype}); {len(clips)} clips; "
          f"text {text_settings()}; fingerprint {fp}", flush=True)

    for system in args.systems:
        out = eval_dir / system
        store = out / "asr_transcripts.jsonl"
        if args.overwrite:
            store.unlink(missing_ok=True)
        elif (out / "asr_metrics.csv").exists():
            print(f"{system}: {out / 'asr_metrics.csv'} exists, skipped (--overwrite to redo)")
            continue
        if system == "clean":
            items = [(clip, None, rows.loc[rows["utterance_id"] == clip, "clean_path"].iloc[0]) for clip in clips]
        elif system == "distorted":
            items = list(zip(rows["utterance_id"], rows["severity"], rows["distorted_path"]))
        else:
            items = [(u, s, audio_dir / system / "audio" / f"{u}_{s}.wav")
                     for u, s in zip(rows["utterance_id"], rows["severity"])]
        missing = [str(p) for _, _, p in items if not Path(p).exists()]
        if missing:
            raise SystemExit(f"{system}: {len(missing)} audio file(s) missing, e.g. {missing[0]}")
        out.mkdir(parents=True, exist_ok=True)
        stored = load_transcripts(store, fp)
        todo = [item for item in items if (item[0], item[1]) not in stored]
        print(f"{system}: {len(items) - len(todo)}/{len(items)} already transcribed, {len(todo)} to go", flush=True)
        start, seconds_audio = time.time(), 0.0
        with open(store, "a", encoding="utf-8") as handle:
            for i in range(0, len(todo), args.batch_size):
                batch = todo[i:i + args.batch_size]
                waves = [load_audio_16k(p) for _, _, p in batch]
                batch_audio = sum(len(w) for w in waves) / 16000
                seconds_audio += batch_audio
                batch_start = time.time()
                texts = asr.transcribe(waves)
                seconds = (time.time() - batch_start) / len(batch)
                for (clip, severity, path), text in zip(batch, texts):
                    record = {"utterance_id": clip, "severity": severity, "audio": str(path), "hypothesis": text,
                              "seconds": round(seconds, 3), "fingerprint": fp,
                              "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stored[(clip, severity)] = record
                handle.flush()
                os.fsync(handle.fileno())
                done = min(i + args.batch_size, len(todo))
                if done % (10 * args.batch_size) < args.batch_size or done == len(todo):
                    rate = (time.time() - start) / done
                    print(f"  {system}: {len(items) - len(todo) + done}/{len(items)} {time.time() - start:.0f}s "
                          f"({rate:.2f} s/clip, ~{rate * (len(todo) - done) / 60:.0f} min left for {system})",
                          flush=True)
        results = []
        for clip, severity, path in items:
            text = stored[(clip, severity)]["hypothesis"]
            results.append({"utterance_id": clip, "severity": severity, "client_id": speaker_of[clip],
                            "audio": str(path), "hypothesis": text, "reference": reference[clip],
                            **error_rates(reference[clip], text)})
        df = pd.DataFrame(results)
        if system == "clean":  # same transcript for every severity, so the table lines up with the others
            df = pd.concat([df.assign(severity=s) for s in SEVERITIES], ignore_index=True)
            df = df.sort_values(["severity", "utterance_id"]).reset_index(drop=True)
            out.mkdir(parents=True, exist_ok=True)
            df[["utterance_id", "client_id", "severity"]].assign(system="clean", timeline="clean").to_csv(
                out / "per_utterance.csv", index=False)
            (out / "run_info.json").write_text(json.dumps({
                "system": "clean", "name": "clean", "split": args.split, "rows": len(df), "timeline": "clean",
                "inference": "the untouched clean clip (ASR reference only)", "git": git_state()}, indent=2))
        df.to_csv(out / "asr_metrics.csv", index=False)
        elapsed = time.time() - start
        per_clip = [r["seconds"] for r in stored.values() if r.get("seconds") is not None]
        info = {"system": system, "rows": len(df), "clips": len(clips), "asr": asr.settings(), "text": text_settings(),
                "fingerprint": fp, "fingerprint_settings": fp_settings, "transcripts": str(store),
                "audio": "clean_path" if system == "clean" else "distorted_path" if system == "distorted"
                else f"{audio_dir / system / 'audio'} (native timeline, resampled to 16 kHz)",
                "reference": "Common Voice sentence (training_manifest transcript)",
                "git": git_state(), "seconds_this_session": round(elapsed, 1),
                "seconds_per_clip": round(float(np.mean(per_clip)), 3) if per_clip else None,
                "real_time_factor_this_session": round(elapsed / seconds_audio, 3) if seconds_audio else None,
                "torch": torch.__version__, "limit": args.limit or None}
        (out / "asr_run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
        by = df.groupby("severity")[["cer_edits", "cer_ref_len", "wer_edits", "wer_ref_len"]].sum()
        print(f"{system}: done ({info['seconds_per_clip']} s/clip); "
              "corpus CER " + ", ".join(f"{s} {r.cer_edits / r.cer_ref_len:.3f}" for s, r in by.iterrows())
              + " | WER " + ", ".join(f"{s} {r.wer_edits / r.wer_ref_len:.3f}" for s, r in by.iterrows()), flush=True)


if __name__ == "__main__":
    main()

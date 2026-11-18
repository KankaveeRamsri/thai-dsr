#!/usr/bin/env python3
"""Evaluate one system on the CV 27.0 test split (308 clips x mild/moderate/severe).

Systems and the timeline each one produces audio on
  distorted        the unprocessed distorted input (floor)              distorted
  copysynth_clean  clean mel -> HiFi-GAN (ceiling for linear models)    clean
  copysynth_tempo  clean mel warped onto the distorted timeline ->      distorted
                   HiFi-GAN (ceiling for tempo models: the exact
                   training target of --alignment tempo, vocoded)
  pv_roundtrip     clean -> per-segment stretch -> inverse stretch,     distorted
                   no vocoder (control: what the inverse warp alone
                   costs every distorted-timeline system)
  copysynth_ref    mel of the stretched clean waveform -> HiFi-GAN      distorted
                   (legacy ceiling of the previous protocol)
  mapper           wav2vec2 layer-L embedding -> mapper -> HiFi-GAN     see below
                   (--checkpoint, --alignment, --layer, default 9; a
                   layer other than 9 adds _layer<NN> to the default name)

Mapper inference (--alignment, plus --linear-length for linear models)
  linear, --linear-length oracle  (default name mapper_linear_oracle)
        the embedding is interpolated to the CLEAN mel frame count, exactly as
        in training. This hands the model the clean duration, which is not
        available in deployment; the output is on the clean timeline.
  linear, --linear-length deploy  (default name mapper_linear_deploy)
        the same model, but the embedding is interpolated to the DISTORTED mel
        frame count, the only length known at deployment; no oracle. The
        output is on the distorted timeline.
  tempo  (default name mapper_tempo)
        the embedding is interpolated only for the frame-rate change, to the
        DISTORTED mel frame count, as in training; no oracle. The output is on
        the distorted timeline.
If a run_info.json (written by notebooks/colab_train.ipynb) or an "alignment"
key sits with the checkpoint, it must agree with --alignment.

Every output is scored against the untouched clean clip. Distorted-timeline
output is brought onto the clean timeline using the logged tempo segments;
those are used for scoring only, never as a system input. Clean-timeline
output is only trimmed/padded to the clean length (< 1 hop, length_residual).

Main comparison (vocoded systems)
  stoi_melwarp / pesq_melwarp  distorted-timeline output is brought back by
                 unwarping its MEL frames onto the clean frame grid (frame
                 interpolation, TempoWarp.unwarp_frames) and vocoding that.
                 Equal to stoi / pesq for clean-timeline systems. Not defined
                 for the floor or pv_roundtrip, which have no mel to vocode.

Secondary (waveform inverse warp; the only one defined for the floor)
  stoi / pesq    distorted-timeline output is trimmed/padded to the distorted
                 length and inverse-warped with TempoWarp.unstretch_waveform
                 (phase vocoder at 1/rate per logged segment). That phase-
                 vocoder pass costs every distorted-timeline system STOI that
                 clean-timeline systems never pay (see pv_roundtrip, and
                 copysynth_tempo vs copysynth_clean), which is why the mel-
                 domain version is the main comparison.

Mel and speaker metrics
  mel_l1         mean |mel - aligned clean target on the system's timeline|:
                 the clean mel (clean timeline) or the clean mel warped onto the
                 distorted timeline (distorted timeline). Mapper: unclamped
                 prediction. Not comparable across timelines.
  mel_l1_clean   mean |mel - clean mel| on the clean frame grid; distorted-
                 timeline mels are first unwarped by frame interpolation.
  spk_sim_same   ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) cosine between
                 the system output (native timeline, 16 kHz) and the clean clip.
  spk_sim_other  the same against every OTHER clean test clip of the same
                 client_id (mean over them), so copying the reference does not
                 help. spk_n_other is how many clips that is.

Pitch / tone metrics (src/evaluation/pitch.py, Praat via parselmouth)
  Computed on the same clean-timeline signal as the main comparison: the
  melwarp audio for vocoded systems (the output itself for clean-timeline
  ones), and the waveform inverse warp for the floor and pv_roundtrip, which
  have no mel. The reference is the untouched clean clip.
  f0_corr        Pearson correlation of F0 (semitones) on frames voiced in both
  f0_rmse_st     RMS F0 error in semitones on frames voiced in both
  vuv_agree      share of frames with the same voiced/unvoiced decision
  f0_n_voiced_both  frames voiced in both (< 10: f0_corr / f0_rmse_st are NaN)
  hnr_db / hnr_clean_db  mean HNR of the output / of the clean clip
  --pitch-only recomputes just these for an existing <out-dir>/<name>/ from its
  saved audio, into pitch_metrics.csv (per_utterance.csv is left untouched);
  the summarizer merges it.

Legacy metrics (kept separate)
  stoi_trunc / pesq_trunc  original protocol (src/evaluation/metrics.py):
                 unwarped clean reference, both truncated to the shorter
                 length, no time alignment.
  stoi_fwd / pesq_fwd      previous aligned protocol: the clean clip stretched
                 forward onto the distorted timeline as the reference.
                 Distorted-timeline systems only (NaN otherwise).

Writes <out-dir>/<name>/per_utterance.csv, run_info.json and the vocoded audio;
the clean-clip ECAPA embeddings are cached in <out-dir>/ecapa_clean_<split>.npz.
scripts/summarize_cv_eval.py builds the tables with confidence intervals.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from pesq import pesq
from pystoi import stoi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.mapper_eval import load_vocoder  # noqa: E402
from src.evaluation.metrics import PESQ_MODE, TARGET_SAMPLE_RATE, compute_pesq, compute_stoi, load_audio_16k  # noqa: E402
from src.evaluation.pitch import PITCH_COLUMNS, analyze, compare  # noqa: E402
from src.inference.run import MEL_CLAMP_MAX, MEL_CLAMP_MIN, get_device, load_mapper  # noqa: E402
from src.utils.config import DEFAULT_MODEL_CONFIG_PATH, DEFAULT_TRAIN_CONFIG_PATH, cv_data_root  # noqa: E402
from src.utils.mel import HIFIGAN_SAMPLE_RATE, compute_mel  # noqa: E402
from src.utils.time_warp import TempoWarp, fit_length, mel_frame_count  # noqa: E402

SEVERITY_ORDER = ["mild", "moderate", "severe"]
SYSTEM_TIMELINE = {"distorted": "distorted", "copysynth_clean": "clean", "copysynth_tempo": "distorted",
                   "pv_roundtrip": "distorted", "copysynth_ref": "distorted"}
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_SAVEDIR = Path("~/.cache/speechbrain/spkrec-ecapa-voxceleb").expanduser()


def safe_pesq(reference, estimate):
    try:
        return float(pesq(TARGET_SAMPLE_RATE, reference, estimate, PESQ_MODE)), ""
    except Exception as exc:  # pesq raises on e.g. no detected utterance
        return float("nan"), str(exc)


def safe_pesq_trunc(reference, estimate):
    try:
        return float(compute_pesq(reference, estimate))
    except Exception:
        return float("nan")


def vocode(vocoder, mel, device, path, keep=True):
    """(T, 80) mel -> HiFi-GAN -> 16-bit wav at ``path`` -> 16 kHz float waveform."""
    voc_in = torch.from_numpy(np.ascontiguousarray(mel.T))[None].float()
    with torch.no_grad():
        wav = vocoder(voc_in.clamp(MEL_CLAMP_MIN, MEL_CLAMP_MAX).to(device)).squeeze().cpu().numpy()
    sf.write(path, np.clip(wav, -1, 1), HIFIGAN_SAMPLE_RATE, subtype="PCM_16")
    audio = load_audio_16k(path)
    if not keep:
        path.unlink()
    return audio


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state():
    def run(*cmd):
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def check_alignment(checkpoint_path, checkpoint, alignment):
    """Fail if anything stored with the checkpoint names a different alignment."""
    sources = {}
    if checkpoint.get("alignment"):
        sources["checkpoint"] = checkpoint["alignment"]
    run_info = Path(checkpoint_path).with_name("run_info.json")
    if run_info.exists():
        sources[str(run_info)] = json.loads(run_info.read_text()).get("config")
    for source, value in sources.items():
        if value is not None and value != alignment:
            raise SystemExit(f"--alignment {alignment} but {source} says {value!r}")
    return sources


class SpeakerEncoder:
    """ECAPA-TDNN speaker embeddings (192-d, L2-normalised) on 16 kHz audio, CPU."""

    def __init__(self):
        # librosa's lazy loader calls inspect.stack(), which walks sys.modules
        # and trips speechbrain's lazy optional-dependency stubs (k2). Load
        # every librosa path this script uses before speechbrain is imported.
        import librosa
        silence = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
        compute_mel(silence, sr=TARGET_SAMPLE_RATE)
        librosa.resample(silence, orig_sr=HIFIGAN_SAMPLE_RATE, target_sr=TARGET_SAMPLE_RATE)
        librosa.effects.time_stretch(silence, rate=1.1)
        from speechbrain.inference.speaker import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(source=ECAPA_SOURCE, savedir=str(ECAPA_SAVEDIR),
                                                    run_opts={"device": "cpu"})
        ckpt = ECAPA_SAVEDIR / "embedding_model.ckpt"  # a symlink into the HF snapshot
        self.revision = Path(os.readlink(ckpt)).parent.name if ckpt.is_symlink() else "unknown"

    @torch.no_grad()
    def __call__(self, wav):
        emb = self.model.encode_batch(torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None])
        emb = emb.reshape(-1).numpy().astype(np.float64)
        return emb / np.linalg.norm(emb)


def clean_speaker_embeddings(path, clips, speaker_of, encoder, clean_paths):
    """ECAPA embeddings of the clean test clips, cached in ``path``."""
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if list(cached["clip_id"]) == clips:
            return dict(zip(clips, cached["embedding"]))
    print(f"computing ECAPA embeddings for {len(clips)} clean clips -> {path}", flush=True)
    embeddings = np.stack([encoder(load_audio_16k(clean_paths[clip])) for clip in clips])
    np.savez(path, clip_id=np.array(clips), client_id=np.array([speaker_of[c] for c in clips]),
             embedding=embeddings, revision=np.array(encoder.revision))
    return dict(zip(clips, embeddings))


class CleanPitch:
    """Pitch tracks of the clean clips, computed once per clip."""

    def __init__(self):
        self.tracks = {}

    def __call__(self, clip, clean):
        if clip not in self.tracks:
            self.tracks[clip] = analyze(clean, TARGET_SAMPLE_RATE)
        return self.tracks[clip]


def pitch_only(out, split_rows, segments):
    """Pitch / tone metrics for an existing system folder, from its saved audio."""
    previous = pd.read_csv(out / "per_utterance.csv")
    info = json.loads((out / "run_info.json").read_text())
    system, timeline = info["system"], info["timeline"]
    by_clip = split_rows.set_index(["utterance_id", "severity"])
    clean_pitch, results, start = CleanPitch(), [], time.time()
    source = ("waveform inverse warp of the distorted input" if system == "distorted" else
              "waveform inverse warp of the stretched clean clip" if system == "pv_roundtrip" else
              "saved output audio (clean timeline)" if timeline == "clean" else "saved audio_melwarp")
    print(f"{out.name}: pitch metrics for {len(previous)} rows from {source}", flush=True)
    for i, row in enumerate(previous.itertuples(), 1):
        paths = by_clip.loc[(row.utterance_id, row.severity)]
        clean = load_audio_16k(paths.clean_path)
        if system in ("distorted", "pv_roundtrip"):
            warp = TempoWarp(segments.loc[(row.utterance_id, row.severity)], TARGET_SAMPLE_RATE)
            native = load_audio_16k(paths.distorted_path) if system == "distorted" else warp.stretch_waveform(clean)
            estimate = warp.unstretch_waveform(fit_length(native, warp.distorted_samples))
        else:
            folder = "audio" if timeline == "clean" else "audio_melwarp"
            estimate = fit_length(load_audio_16k(out / folder / f"{row.utterance_id}_{row.severity}.wav"), len(clean))
        metrics = compare(clean_pitch(row.utterance_id, clean), analyze(estimate, TARGET_SAMPLE_RATE))
        results.append({"utterance_id": row.utterance_id, "severity": row.severity, **metrics})
        if i % 100 == 0 or i == len(previous):
            print(f"  {i}/{len(previous)} {time.time() - start:.0f}s", flush=True)
    df = pd.DataFrame(results)
    df.to_csv(out / "pitch_metrics.csv", index=False)
    (out / "pitch_run_info.json").write_text(json.dumps({
        "system": system, "timeline": timeline, "rows": len(df), "estimate": source,
        "reference": "untouched clean clip", "git": git_state(), "seconds": round(time.time() - start, 1),
        "f0_metrics_nan": int(df["f0_corr"].isna().sum())}, indent=2))
    print(df.merge(previous[["utterance_id", "severity"]]).groupby("severity")[PITCH_COLUMNS].mean().round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=(*SYSTEM_TIMELINE, "mapper"), required=True)
    parser.add_argument("--checkpoint", help="mapper checkpoint (for --system mapper)")
    parser.add_argument("--alignment", choices=("linear", "tempo"),
                        help="what the mapper was trained with (for --system mapper)")
    parser.add_argument("--linear-length", choices=("oracle", "deploy"), default="oracle",
                        help="linear models only: interpolate the embedding to the clean (oracle) or the "
                             "distorted (deploy) mel frame count")
    parser.add_argument("--name", help="output folder name (default: the system name, mapper_tempo, "
                                       "or mapper_linear_<linear-length>)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="default: <data root>/training_manifest.csv")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/manifests/cv27_th_splits.json")
    parser.add_argument("--speakers", type=Path, default=ROOT / "data/manifests/cv27_th_speaker_split.csv")
    parser.add_argument("--params", type=Path, default=ROOT / "data/manifests/cv27_th_distortion_params.csv")
    parser.add_argument("--embeddings-dir", type=Path, default=None,
                        help="default: <data root>/embeddings")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <data root>/eval/test_full")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--limit", type=int, default=0, help="only the first N clips per severity (smoke test)")
    parser.add_argument("--device", default=None, help="mapper/vocoder device (default: cuda, then mps, then cpu)")
    parser.add_argument("--no-audio", action="store_true", help="do not keep the output wav files")
    parser.add_argument("--pitch-only", action="store_true",
                        help="only (re)compute the pitch / tone metrics of an existing <out-dir>/<name>/ "
                             "from its saved audio, into pitch_metrics.csv")
    args = parser.parse_args()
    if args.system == "mapper" and not args.pitch_only and not (args.checkpoint and args.alignment):
        parser.error("--system mapper needs --checkpoint and --alignment")
    if args.system == "mapper" and args.pitch_only and not (args.name or args.alignment):
        parser.error("--pitch-only with --system mapper needs --name or --alignment")

    root = cv_data_root()
    manifest = pd.read_csv(args.manifest or root / "training_manifest.csv")
    ids = json.loads(args.splits.read_text(encoding="utf-8"))["splits"][args.split]
    split_rows = manifest[manifest["utterance_id"].isin(set(ids))].sort_values(["severity", "utterance_id"])
    rows = split_rows
    if args.limit:
        rows = rows.groupby("severity", group_keys=False).head(args.limit)
    speakers = pd.read_csv(args.speakers, usecols=["clip_id", "client_id"])
    speaker_of = dict(zip(speakers["clip_id"], speakers["client_id"]))
    segments = pd.read_csv(args.params).set_index(["clip_id", "severity"])["tempo_segments"]
    embeddings_dir = args.embeddings_dir or root / "embeddings"
    out_root = args.out_dir or root / "eval" / "test_full"
    variant = f"linear_{args.linear_length}" if args.alignment == "linear" else args.alignment
    checkpoint = None
    if args.system == "mapper" and args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint is not None and "input_mean" in checkpoint["model_state_dict"]:
        # ablation recipe (input normalisation, docs/ablation_plan.md Amendment 1): every layer is named
        layer_suffix = f"_norm_layer{args.layer:02d}"
    else:  # W2 recipe: layer 9 keeps the plain name
        layer_suffix = "" if args.layer == 9 else f"_layer{args.layer:02d}"
    name = args.name or (f"mapper_{variant}{layer_suffix}" if args.system == "mapper" else args.system)
    out = out_root / name
    if args.pitch_only:
        pitch_only(out, split_rows, segments)
        return
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "audio_melwarp").mkdir(exist_ok=True)

    timeline = SYSTEM_TIMELINE.get(args.system) or (
        "clean" if variant == "linear_oracle" else "distorted")
    device = torch.device(args.device) if args.device else get_device()
    vocoder = load_vocoder(device) if args.system not in ("distorted", "pv_roundtrip") else None
    mapper, checkpoint_info = None, {}
    if args.system == "mapper":
        sources = check_alignment(args.checkpoint, checkpoint, args.alignment)
        run_info = Path(args.checkpoint).with_name("run_info.json")
        stored_layers = {"checkpoint": checkpoint.get("wav2vec2_layer"),
                         str(run_info): json.loads(run_info.read_text()).get("layer", 9) if run_info.exists() else None}
        for source, value in stored_layers.items():
            if value is not None and int(value) != args.layer:
                raise SystemExit(f"--layer {args.layer} but {source} says layer {value}")
        mapper = load_mapper(args.checkpoint, device, DEFAULT_MODEL_CONFIG_PATH,
                             DEFAULT_TRAIN_CONFIG_PATH, layer=args.layer)
        checkpoint_info = {"path": str(Path(args.checkpoint).resolve()), "sha256": sha256(args.checkpoint),
                           "epoch": checkpoint.get("epoch"), "best_epoch": checkpoint.get("best_epoch"),
                           "val_loss": checkpoint.get("val_loss"), "alignment": args.alignment,
                           "linear_length": args.linear_length if args.alignment == "linear" else None,
                           "input_norm": checkpoint.get("input_norm") or (
                               "input_mean" in checkpoint["model_state_dict"]),
                           "alignment_confirmed_by": sources or "--alignment only (nothing stored with the checkpoint)"}
        del checkpoint

    encoder = SpeakerEncoder()
    clean_pitch = CleanPitch()
    # Clean references cover the whole split, also under --limit.
    test_clips = sorted(set(split_rows["utterance_id"]))
    clean_paths = dict(zip(split_rows["utterance_id"], split_rows["clean_path"]))
    clean_spk = clean_speaker_embeddings(out_root / f"ecapa_clean_{args.split}.npz", test_clips,
                                         speaker_of, encoder, clean_paths)
    by_speaker = {}
    for clip in test_clips:
        by_speaker.setdefault(speaker_of[clip], []).append(clip)

    print(f"{name}: {len(rows)} rows ({args.split}), {timeline} timeline, {device} -> {out}", flush=True)
    results, start = [], time.time()
    for i, row in enumerate(rows.itertuples(), 1):
        clean = load_audio_16k(row.clean_path)
        distorted = load_audio_16k(row.distorted_path)
        warp = TempoWarp(segments.loc[(row.utterance_id, row.severity)], TARGET_SAMPLE_RATE)
        n_clean = mel_frame_count(len(clean), TARGET_SAMPLE_RATE)
        n_dist = mel_frame_count(len(distorted), TARGET_SAMPLE_RATE)
        clean_mel = compute_mel(clean, sr=TARGET_SAMPLE_RATE).T
        assert len(clean_mel) == n_clean
        tempo_target = warp.warp_frames(clean_mel, n_target=n_dist)
        target = clean_mel if timeline == "clean" else tempo_target
        reference_fwd = None
        if timeline == "distorted":
            reference_fwd = warp.stretch_waveform(clean)
            if len(reference_fwd) != len(distorted):
                raise ValueError(f"{row.utterance_id}: aligned reference length mismatch")

        mel = None
        if args.system == "distorted":
            mel, wav = compute_mel(distorted, sr=TARGET_SAMPLE_RATE).T, distorted
        elif args.system == "pv_roundtrip":
            mel, wav = compute_mel(reference_fwd, sr=TARGET_SAMPLE_RATE).T, reference_fwd
        else:
            if args.system == "copysynth_clean":
                mel = clean_mel
            elif args.system == "copysynth_tempo":
                mel = tempo_target
            elif args.system == "copysynth_ref":
                mel = compute_mel(reference_fwd, sr=TARGET_SAMPLE_RATE).T
            else:
                n_frames = n_clean if timeline == "clean" else n_dist
                emb = torch.from_numpy(np.load(
                    embeddings_dir / f"{row.utterance_id}_{row.severity}_layer{args.layer:02d}.npy")).float()
                emb = F.interpolate(emb.T[None], size=n_frames, mode="linear", align_corners=False)[0].T
                with torch.no_grad():
                    pred = mapper(emb[None].to(device), torch.tensor([n_frames], device=device))
                mel = pred[0].cpu().numpy()
            wav = vocode(vocoder, mel, device, out / "audio" / f"{row.utterance_id}_{row.severity}.wav",
                         keep=not args.no_audio)

        native = len(clean) if timeline == "clean" else len(distorted)
        on_native = fit_length(wav, native)
        restored = on_native if timeline == "clean" else warp.unstretch_waveform(on_native)
        pesq_value, pesq_err = safe_pesq(clean, restored)
        mel_clean_grid = None if mel is None else (mel if timeline == "clean" else warp.unwarp_frames(mel, n_clean))

        spk = encoder(wav)
        others = [c for c in by_speaker[speaker_of[row.utterance_id]] if c != row.utterance_id]
        result = {
            "utterance_id": row.utterance_id, "client_id": speaker_of[row.utterance_id],
            "severity": row.severity, "system": name, "timeline": timeline,
            "stoi": float(stoi(clean, restored, TARGET_SAMPLE_RATE, extended=False)),
            "pesq": pesq_value,
            "mel_l1": float(np.abs(mel - target).mean()) if mel is not None else float("nan"),
            "mel_l1_clean": float(np.abs(mel_clean_grid - clean_mel).mean()) if mel is not None else float("nan"),
            "spk_sim_same": float(spk @ clean_spk[row.utterance_id]),
            "spk_sim_other": float(np.mean([spk @ clean_spk[c] for c in others])) if others else float("nan"),
            "spk_n_other": len(others),
            "stoi_trunc": float(compute_stoi(clean, wav)),
            "pesq_trunc": safe_pesq_trunc(clean, wav),
            "stoi_fwd": float("nan"), "pesq_fwd": float("nan"),
            "stoi_melwarp": float("nan"), "pesq_melwarp": float("nan"),
            "length_residual": len(wav) - native, "pesq_error": pesq_err,
        }
        if timeline == "distorted":
            result["stoi_fwd"] = float(stoi(reference_fwd, on_native, TARGET_SAMPLE_RATE, extended=False))
            result["pesq_fwd"] = safe_pesq(reference_fwd, on_native)[0]
        if vocoder is not None:
            if timeline == "clean":  # nothing to unwarp: identical to the primary metric
                melwarp = restored
            else:
                melwarp = fit_length(vocode(vocoder, mel_clean_grid, device,
                                            out / "audio_melwarp" / f"{row.utterance_id}_{row.severity}.wav",
                                            keep=not args.no_audio), len(clean))
            result["stoi_melwarp"] = float(stoi(clean, melwarp, TARGET_SAMPLE_RATE, extended=False))
            result["pesq_melwarp"] = safe_pesq(clean, melwarp)[0]
        # pitch / tone on the same clean-timeline signal as the main comparison
        result.update(compare(clean_pitch(row.utterance_id, clean),
                              analyze(melwarp if vocoder is not None else restored, TARGET_SAMPLE_RATE)))
        results.append(result)
        if i % 50 == 0 or i == len(rows):
            elapsed = time.time() - start
            print(f"  {i}/{len(rows)} {elapsed:.0f}s ({elapsed / i:.2f}s/row)", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(out / "per_utterance.csv", index=False)
    info = {
        "system": args.system, "name": name, "split": args.split, "rows": len(df),
        "timeline": timeline,
        "inference": {
            "distorted": "the distorted input itself",
            "copysynth_clean": "clean mel -> HiFi-GAN, on the clean timeline",
            "copysynth_tempo": "clean mel warped onto the distorted timeline -> HiFi-GAN",
            "pv_roundtrip": "clean stretched onto the distorted timeline (phase vocoder), no vocoder",
            "copysynth_ref": "mel of the stretched clean waveform -> HiFi-GAN",
        }.get(args.system) or {
            "linear_oracle": "linear model; embedding interpolated to the CLEAN mel frame count, as in "
                             "training (uses the clean duration as an oracle)",
            "linear_deploy": "linear model; embedding interpolated to the DISTORTED mel frame count "
                             "(no oracle; differs from how it was trained)",
            "tempo": "tempo model; embedding interpolated to the DISTORTED mel frame count, as in training",
        }[variant],
        "reference": "untouched clean clip; " + (
            "output trimmed/padded to the clean length (all metrics)" if timeline == "clean" else
            "main *_melwarp: mel frames unwarped onto the clean grid, then vocoded; "
            "secondary stoi/pesq: waveform trimmed/padded to the distorted length, then inverse-warped "
            "(TempoWarp.unstretch_waveform)"),
        "checkpoint": checkpoint_info or None, "layer": args.layer if args.system == "mapper" else None,
        "device": str(device), "ecapa": {"source": ECAPA_SOURCE, "revision": encoder.revision},
        "git": git_state(), "seconds": round(time.time() - start, 1),
        "max_abs_length_residual": int(df["length_residual"].abs().max()),
        "pesq_failed": int(df["pesq"].isna().sum()),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    metrics = ["stoi_melwarp", "pesq_melwarp", "stoi", "pesq", "mel_l1", "mel_l1_clean",
               "spk_sim_same", "spk_sim_other", "f0_corr", "f0_rmse_st", "vuv_agree", "hnr_db", "stoi_trunc", "pesq_trunc", "stoi_fwd", "pesq_fwd"]
    means = df.groupby("severity")[metrics].mean().reindex([s for s in SEVERITY_ORDER if s in set(df["severity"])])
    print(f"\n{name} ({timeline} timeline); max |length residual| {info['max_abs_length_residual']} samples, "
          f"{info['pesq_failed']} PESQ failure(s), {info['seconds']:.0f}s")
    print(means.round(4).to_string())


if __name__ == "__main__":
    main()

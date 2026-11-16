#!/usr/bin/env python3
"""Exploratory check of real dysarthric recordings against the simulation and the model.

Not a research-grade evaluation: it is for 1-2 recordings from a real speaker.
For each recording it
  1. converts the audio to 16 kHz mono WAV,
  2. measures jitter / shimmer / HNR / F0 / speaking rate (Praat, same calls as
     scripts/measure_acoustics.py) next to our clean and mild/moderate/severe
     simulated versions of the same sentence,
  3. runs the baseline pipeline (wav2vec2 layer 9 -> mapper_layer9.pt -> HiFi-GAN
     UNIVERSAL_V1) on the recording and on our simulated severe version,
  4. transcribes everything with the Thai wav2vec2 CTC model and reports CER
     against the known sentence.

STOI/PESQ are not computed: a real speaker has no clean reference recording.
The ASR model shares its lower layers with our encoder, so CER is a rough
indicator only.

Usage:
  python scripts/patient_probe.py --item P01_s1.m4a 002 --item P01_s2.m4a 004 --name P01
  python scripts/patient_probe.py --item P01_aa.wav my_clean_aa.wav --name P01
The reference is a data/raw utterance id prefix (002, 004, ...), a clean WAV of
the same content (distorted on the fly with seed 42), or "-" for none.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from measure_acoustics import FEATURES, measure_acoustics  # noqa: E402
from src.inference.run import load_hifigan, load_mapper, reconstruct  # noqa: E402
from src.models.encoder import Wav2Vec2ContentEncoder  # noqa: E402
from src.preprocessing.distortion import SEVERITY_LEVELS, apply_distortion  # noqa: E402
from src.utils.config import DEFAULT_MODEL_CONFIG_PATH, DEFAULT_TRAIN_CONFIG_PATH, load_yaml_config  # noqa: E402

MAPPER = ROOT / 'results/checkpoints/mapper_layer9.pt'
VOCODER_DIR = ROOT / 'vendor/hifi-gan/checkpoints/UNIVERSAL_V1'
OUT_ROOT = ROOT / 'results/patient_probe'


def to_wav16k(src, dest):
    """Convert any audio file ffmpeg can read to 16 kHz mono PCM WAV."""
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src), '-ac', '1', '-ar', '16000',
                    '-c:a', 'pcm_s16le', str(dest)], check=True)
    return dest


def resolve_reference(ref, workdir):
    """Return (transcript or None, {condition: path}) for the reference spec."""
    if ref == '-':
        return None, {}
    hits = sorted((ROOT / 'data/raw').glob(f'{ref}_*.wav')) if not ref.lower().endswith('.wav') else []
    if hits:
        clean = hits[0]
        stem = clean.stem
        conditions = {'clean': clean}
        for sev in SEVERITY_LEVELS:
            conditions[sev] = ROOT / 'data/distorted' / f'{stem}_{sev}.wav'
        return stem.split('_', 1)[1], conditions
    clean = Path(ref)
    if not clean.is_file():
        raise SystemExit(f'Reference not found: {ref} (use a data/raw id like 002, a WAV path, or -)')
    import librosa
    y, sr = librosa.load(clean, sr=None, mono=True)
    conditions = {'clean': clean}
    random.seed(42)
    np.random.seed(42)
    for sev in SEVERITY_LEVELS:
        dest = workdir / f'ref_{clean.stem}_{sev}.wav'
        sf.write(dest, apply_distortion(y, sr, sev), sr)
        conditions[sev] = dest
    return None, conditions


def cer(reference, hypothesis):
    """Character error rate, ignoring spaces (Thai has no word spacing)."""
    r, h = reference.replace(' ', ''), hypothesis.replace(' ', '')
    if not r:
        return float('nan')
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


class ThaiASR:
    def __init__(self, name, device):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.processor = Wav2Vec2Processor.from_pretrained(name)
        self.model = Wav2Vec2ForCTC.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, wav_path):
        import librosa
        audio, _ = librosa.load(wav_path, sr=16000, mono=True)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors='pt')
        logits = self.model(inputs.input_values.to(self.device)).logits
        return self.processor.batch_decode(logits.argmax(-1))[0].strip()


def fmt(v, digits=2):
    return '-' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{digits}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--item', nargs=2, action='append', required=True, metavar=('AUDIO', 'REFERENCE'))
    p.add_argument('--name', required=True, help='anonymous speaker code, e.g. P01 (never a real name)')
    p.add_argument('--no-asr', action='store_true', help='skip transcription')
    args = p.parse_args()

    out = OUT_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cpu')
    model_name = load_yaml_config(DEFAULT_MODEL_CONFIG_PATH)['encoder']['content_model']

    print('Loading encoder, mapper and vocoder ...', flush=True)
    encoder = Wav2Vec2ContentEncoder(device=device, layer=9)
    mapper = load_mapper(str(MAPPER), device, DEFAULT_MODEL_CONFIG_PATH, DEFAULT_TRAIN_CONFIG_PATH, layer=9)
    vocoder, vocoder_config = load_hifigan(str(VOCODER_DIR), device)
    asr = None if args.no_asr else ThaiASR(model_name, device)

    report = dict(speaker=args.name, mapper=str(MAPPER.relative_to(ROOT)), asr_model=None if asr is None else model_name,
                  note='exploratory; no clean reference for the real speaker, so no STOI/PESQ', items=[])
    for k, (audio, ref) in enumerate(args.item, 1):
        tag = f'{args.name}_{k:02d}'
        patient = to_wav16k(audio, out / f'{tag}_input.wav')
        transcript, refs = resolve_reference(ref, out)
        rows = {'real (input)': patient}
        rows.update({f'sim {c}': path for c, path in refs.items()})
        rows['real -> model'] = Path(reconstruct(str(patient), str(out), encoder, mapper, vocoder, vocoder_config, device))
        if 'severe' in refs:
            sim_out = reconstruct(str(refs['severe']), str(out), encoder, mapper, vocoder, vocoder_config, device)
            rows['sim severe -> model'] = Path(sim_out)

        item = dict(input=Path(audio).name, reference=ref, transcript=transcript, conditions={})
        for label, path in rows.items():
            entry = {'file': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)}
            entry.update(measure_acoustics(str(path)))
            if asr is not None and transcript:
                entry['asr'] = asr(path)
                entry['cer'] = cer(transcript, entry['asr'])
            item['conditions'][label] = entry
        acoustic_rows = {k: v for k, v in item['conditions'].items() if '-> model' not in k}
        sims = {k: v for k, v in acoustic_rows.items() if k.startswith('sim ')}
        if sims:
            real = item['conditions']['real (input)']
            item['closest_simulated_level'] = {
                f: min(sims, key=lambda s: abs(sims[s][f] - real[f]) if not np.isnan(sims[s][f]) and not np.isnan(real[f]) else np.inf)
                for f in FEATURES}
        report['items'].append(item)

    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    lines = [f'# Patient probe: {args.name}', '', report['note'], '']
    for item in report['items']:
        lines += [f'## {item["input"]}  (reference: {item["reference"]})', '']
        if item['transcript']:
            lines += [f'Sentence: {item["transcript"]}', '']
        lines += ['| condition | jitter % | shimmer % | HNR dB | F0 Hz | voiced/s | ASR | CER |', '|---|---|---|---|---|---|---|---|']
        for label, e in item['conditions'].items():
            lines.append(f'| {label} | {fmt(e["jitter_local_pct"])} | {fmt(e["shimmer_local_pct"])} | {fmt(e["hnr_db"])} | '
                         f'{fmt(e["mean_f0_hz"], 0)} | {fmt(e["voiced_rate_per_sec"], 1)} | {e.get("asr", "-")} | {fmt(e.get("cer"))} |')
        if item.get('closest_simulated_level'):
            lines += ['', 'Closest simulated level per feature: ' +
                      ', '.join(f'{f}: {lvl}' for f, lvl in item['closest_simulated_level'].items())]
        lines.append('')
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    print('Wrote', out / 'report.md')


if __name__ == '__main__':
    main()

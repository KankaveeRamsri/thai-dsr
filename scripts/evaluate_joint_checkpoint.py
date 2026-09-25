"""Real three-way full-pipeline TEST evaluation of an explicitly selected joint snapshot."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_vocoder_finetune import select_rows, METRICS, digest
from src.evaluation.metrics import load_audio_16k
from src.inference.run import (
    AttrDict, Generator, Wav2Vec2ContentEncoder, load_mapper,
    compute_mel_frame_count, interpolate_embedding, MEL_CLAMP_MIN, MEL_CLAMP_MAX,
)


def evaluate(joint_checkpoint, output, device):
    joint_checkpoint, output = Path(joint_checkpoint).resolve(), Path(output).resolve()
    if not joint_checkpoint.is_file():
        raise FileNotFoundError(joint_checkpoint)
    # Never mix or overwrite the previous experiment's generated files.
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    mapper_path = ROOT / 'results/checkpoints/mapper_layer9.pt'
    variants = {
        'baseline': ROOT / 'vendor/hifi-gan/checkpoints/UNIVERSAL_V1/g_02500000',
        'approach_a': ROOT / 'results/checkpoints/hifigan_thai/g_00010000',
        'joint_finetuned': joint_checkpoint,
    }
    configs = {name: json.loads(p.with_name('config.json').read_text())
               for name, p in variants.items() if name != 'joint_finetuned'}
    assert configs['baseline'] == configs['approach_a']
    sample_rate = int(configs['baseline']['sampling_rate'])
    rows = select_rows()
    assert len(rows) == 8
    provenance = dict(joint_checkpoint=str(joint_checkpoint), joint_sha256=digest(joint_checkpoint),
                      original_mapper=str(mapper_path), original_mapper_sha256=digest(mapper_path),
                      device=str(device), utterances=rows,
                      checkpoints={k: dict(path=str(p), sha256=digest(p)) for k, p in variants.items()},
                      protocol='Saved PCM_16 -> 16 kHz; shared-length truncation; no gain/time alignment')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False)+'\n')
    encoder = Wav2Vec2ContentEncoder(device=device, layer=9)
    encoder.model.eval().requires_grad_(False)
    references = {}
    for row in rows:
        folder = output / row['utterance_id']
        folder.mkdir()
        clean, sr = sf.read(ROOT / row['clean_path'], dtype='float32', always_2d=True)
        clean = clean.mean(axis=1)
        if sr != sample_rate:
            clean = librosa.resample(clean, orig_sr=sr, target_sr=sample_rate)
        sf.write(folder/'ground_truth.wav', clean, sample_rate, subtype='PCM_16')
        references[row['utterance_id']] = load_audio_16k(folder/'ground_truth.wav')
    results = []
    with (output/'results.csv').open('w', newline='') as handle, tqdm(total=24, unit='run') as progress:
        writer = csv.DictWriter(handle, fieldnames=['utterance_id', 'variant', 'stoi', 'pesq', 'snr'])
        writer.writeheader()
        for variant, checkpoint in variants.items():
            state = torch.load(checkpoint, map_location='cpu', weights_only=True,
                               mmap=(variant == 'joint_finetuned'))
            config = dict(configs['baseline'])
            mapper = load_mapper(mapper_path, device, ROOT/'configs/model.yaml', ROOT/'configs/train.yaml', layer=9)
            if variant == 'joint_finetuned':
                mapper.load_state_dict(state['mapper'], strict=True)
                config = dict(state.get('vocoder_config', config))
                assert int(config['sampling_rate']) == sample_rate
                print('Joint checkpoint step:', state['step'], flush=True)
            generator = Generator(AttrDict(config)).to(device)
            generator.load_state_dict(state['generator'], strict=True)
            del state
            mapper.eval().requires_grad_(False)
            generator.eval().requires_grad_(False)
            generator.remove_weight_norm()
            for row in rows:
                uid = row['utterance_id']
                progress.set_postfix(variant=variant, utterance=uid)
                distorted = ROOT / row['distorted_path']
                with torch.inference_mode():
                    embedding = encoder.encode(str(distorted))
                    frames = compute_mel_frame_count(str(distorted))
                    inputs = interpolate_embedding(embedding, frames).unsqueeze(0).to(device)
                    mel = mapper(inputs).clamp(MEL_CLAMP_MIN, MEL_CLAMP_MAX).transpose(1, 2)
                    audio = generator(mel).squeeze().cpu().numpy().astype(np.float32)
                if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                    raise RuntimeError(f'Invalid generated waveform: {variant}/{uid}')
                wav = output / uid / f'{variant}.wav'
                sf.write(wav, np.clip(audio, -1, 1), sample_rate, subtype='PCM_16')
                estimate = load_audio_16k(wav)
                scores = {k: float(fn(references[uid], estimate)) for k, fn in METRICS.items()}
                if not all(np.isfinite(v) for v in scores.values()):
                    raise RuntimeError(f'Non-finite metrics: {variant}/{uid}: {scores}')
                result = dict(utterance_id=uid, variant=variant, **scores)
                results.append(result)
                writer.writerow(result)
                handle.flush()
                progress.update(1)
                del embedding, inputs, mel, audio
            del mapper, generator
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    assert len(results) == 24
    summary = {}
    print('\nVariant            STOI mean +/- std       PESQ mean +/- std       SNR dB mean +/- std')
    for variant in variants:
        subset = [r for r in results if r['variant'] == variant]
        summary[variant] = {k: dict(mean=float(np.mean([r[k] for r in subset])),
                                    std=float(np.std([r[k] for r in subset], ddof=1))) for k in METRICS}
        print(f'{variant:<18} ' + '   '.join(f'{v["mean"]:.6f} +/- {v["std"]:.6f}' for v in summary[variant].values()))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='Must be a NEW directory')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cuda')
    args = parser.parse_args()
    evaluate(args.joint_checkpoint, args.output_dir, torch.device(args.device))

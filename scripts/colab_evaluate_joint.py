# Paste this entire file into a Colab cell after mounting Drive and unpacking the bundle.
from pathlib import Path
import csv
import gc
import json
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm

ROOT = Path.cwd().resolve()  # Run from the cloned thai-dsr repo root.
assert (ROOT / 'src/inference/run.py').is_file(), 'Change CWD to the thai-dsr repo.'
sys.path.insert(0, str(ROOT))
from scripts.evaluate_vocoder_finetune import select_rows, METRICS
from src.evaluation.metrics import load_audio_16k
from src.inference.run import (
    AttrDict, Generator, Wav2Vec2ContentEncoder, load_mapper,
    compute_mel_frame_count, interpolate_embedding, MEL_CLAMP_MIN, MEL_CLAMP_MAX,
)

CHECKPOINT_ROOT = Path(globals().get(
    'CHECKPOINT_ROOT', '/content/drive/MyDrive/thai_dsr_colab/checkpoints'))
OUTPUT = Path('/content/drive/MyDrive/thai_dsr_colab/eval_joint')
assert Path('/content/drive/MyDrive').is_dir(), 'Mount Google Drive first.'
assert torch.cuda.is_available(), 'Select a CUDA GPU runtime.'
device = torch.device('cuda')

candidates = [p for p in CHECKPOINT_ROOT.glob('full_*/joint_*.pt')
              if p.is_file() and p.stem.removeprefix('joint_').isdigit()]
if not candidates:
    raise FileNotFoundError(f'No full_*/joint_*.pt under {CHECKPOINT_ROOT}')
joint_path = max(candidates, key=lambda p: (int(p.stem.split('_')[-1]), p.stat().st_mtime))
original_mapper = ROOT / 'results/checkpoints/mapper_layer9.pt'
variants = {
    'baseline': ROOT / 'vendor/hifi-gan/checkpoints/UNIVERSAL_V1/g_02500000',
    'approach_a': ROOT / 'results/checkpoints/hifigan_thai/g_00010000',
    'joint_finetuned': joint_path,
}
configs = {name: json.loads(path.with_name('config.json').read_text())
           for name, path in variants.items() if name != 'joint_finetuned'}
assert configs['baseline'] == configs['approach_a'], 'Vocoder configs differ.'
rows = select_rows()  # Exactly the same eight severe TEST utterances as the existing eval.
assert len(rows) == 8 and len({r['utterance_id'] for r in rows}) == 8
for path in [original_mapper, *variants.values()]:
    assert path.is_file(), path
for row in rows:
    for key in ('clean_path', 'distorted_path'):
        assert (ROOT / row[key]).is_file(), row[key]
sample_rate = int(configs['baseline']['sampling_rate'])
OUTPUT.mkdir(parents=True, exist_ok=True)
print('GPU:', torch.cuda.get_device_name(0))
print('Joint checkpoint:', joint_path)
print('TEST utterances:', [r['utterance_id'] for r in rows])


def evaluate_three_variants():
    encoder = Wav2Vec2ContentEncoder(device=device, layer=9)
    encoder.model.eval().requires_grad_(False)
    references = {}
    for row in rows:
        folder = OUTPUT / row['utterance_id']
        folder.mkdir(parents=True, exist_ok=True)
        clean, sr = sf.read(ROOT / row['clean_path'], dtype='float32', always_2d=True)
        clean = clean.mean(axis=1)
        if sr != sample_rate:
            clean = librosa.resample(clean, orig_sr=sr, target_sr=sample_rate)
        sf.write(folder / 'ground_truth.wav', clean, sample_rate, subtype='PCM_16')
        references[row['utterance_id']] = load_audio_16k(folder / 'ground_truth.wav')

    results = []
    columns = ['utterance_id', 'variant', 'stoi', 'pesq', 'snr']
    with (OUTPUT / 'results.csv').open('w', newline='', encoding='utf-8') as handle, \
            tqdm(total=24, desc='Real reconstruction + metrics', unit='run') as progress:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for variant, checkpoint_path in variants.items():
            # CPU mmap avoids moving the joint discriminator/optimizer states to CUDA.
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True,
                               mmap=(variant == 'joint_finetuned'))
            config = dict(configs['baseline'])
            mapper = load_mapper(original_mapper, device, ROOT / 'configs/model.yaml',
                                 ROOT / 'configs/train.yaml', layer=9)
            if variant == 'joint_finetuned':
                assert {'mapper', 'generator', 'step'}.issubset(state), 'Invalid joint checkpoint.'
                assert int(state['step']) == int(checkpoint_path.stem.split('_')[-1])
                config = dict(state.get('vocoder_config', config))
                assert int(config['sampling_rate']) == sample_rate
                mapper.load_state_dict(state['mapper'], strict=True)
            # A NEW generator is essential: load weight-normalized weights BEFORE removal.
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
                # Actual full forward pipeline for EACH of the 24 evaluations.
                with torch.inference_mode():
                    embedding = encoder.encode(str(distorted))
                    frames = compute_mel_frame_count(str(distorted))
                    inputs = interpolate_embedding(embedding, frames).unsqueeze(0).to(device)
                    mel = mapper(inputs).clamp(MEL_CLAMP_MIN, MEL_CLAMP_MAX).transpose(1, 2)
                    audio = generator(mel).squeeze().cpu().numpy().astype(np.float32)
                if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                    raise RuntimeError(f'Invalid generated waveform: {variant}/{uid}')
                wav_path = OUTPUT / uid / f'{variant}.wav'
                sf.write(wav_path, np.clip(audio, -1, 1), sample_rate, subtype='PCM_16')
                # Score the actual SAVED PCM_16 output, exactly as the original evaluator.
                estimate = load_audio_16k(wav_path)
                scores = {name: float(function(references[uid], estimate))
                          for name, function in METRICS.items()}
                if not all(np.isfinite(value) for value in scores.values()):
                    raise RuntimeError(f'Non-finite metrics: {variant}/{uid}: {scores}')
                result = dict(utterance_id=uid, variant=variant, **scores)
                writer.writerow(result)
                handle.flush()
                results.append(result)
                progress.update(1)
                del inputs, mel, embedding, audio
            del mapper, generator
            gc.collect()
            torch.cuda.empty_cache()
    return pd.DataFrame(results, columns=columns)


results = evaluate_three_variants()
assert len(results) == 24 and not results.duplicated(['utterance_id', 'variant']).any()
summary = []
for variant in variants:
    subset = results[results['variant'] == variant]
    assert len(subset) == 8
    summary.append({'variant': variant, **{
        metric: f'{subset[metric].mean():.6f} +/- {subset[metric].std(ddof=1):.6f}'
        for metric in METRICS}})
print('\nMean +/- sample std across 8 TEST utterances (SNR in dB):')
print(pd.DataFrame(summary).set_index('variant').to_string())
print('\nProtocol: saved PCM_16 -> 16 kHz; shared-length truncation; no gain/time alignment.')
print('CSV:', OUTPUT / 'results.csv')
print('Generated WAVs and clean references:', OUTPUT)

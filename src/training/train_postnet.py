"""Train ONLY a fresh residual postnet on cached frozen-mapper outputs. No GAN."""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
from datetime import datetime

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from src.models.postnet import Postnet
from src.inference.run import (load_mapper, Wav2Vec2ContentEncoder, Generator, AttrDict,
                               interpolate_embedding, compute_mel_frame_count, MEL_CLAMP_MIN, MEL_CLAMP_MAX)
from src.utils.mel import compute_mel
from src.evaluation.metrics import compute_stoi, compute_pesq, load_audio_16k

ROOT = Path(__file__).resolve().parents[2]


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT/p


def digest(file):
    h = hashlib.sha256()
    with open(file, 'rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def append_record(file, record):
    with file.open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    print(file.stem.upper()+' '+json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def pairs(manifest, splits_file):
    splits = json.loads(path(splits_file).read_text())['splits']
    ids = [uid for split in ('train', 'val', 'test') for uid in splits[split]]
    if len(ids) != len(set(ids)):
        raise ValueError('Split leakage/duplicate IDs')
    with path(manifest).open() as f:
        rows = [r for r in csv.DictReader(f) if r['severity'] == 'severe']
    by_id = {r['utterance_id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError('Severe manifest rows must match split IDs exactly')
    return {s: [by_id[uid] for uid in sorted(splits[s])] for s in splits}


def mel_l1(refined, target):
    # The frozen mapper and postnet ALWAYS use inference/distorted duration.
    # Only the loss view is linearly resized to clean mel length. This corrects
    # global length only; it does not claim local phonetic/tempo alignment.
    if refined.shape[-1] != target.shape[-1]:
        refined = F.interpolate(refined, size=target.shape[-1], mode='linear', align_corners=False)
    return F.l1_loss(refined, target)


@torch.no_grad()
def build_cache(rows, mapper, encoder, device, out):
    cache = {}
    (out/'cache').mkdir()
    for index, row in enumerate(rows, 1):
        uid = row['utterance_id']
        distorted = path(row['distorted_path'])
        embedding = encoder.encode(str(distorted))
        frames = compute_mel_frame_count(str(distorted))
        inputs = interpolate_embedding(embedding, frames).unsqueeze(0).to(device)
        predicted = mapper(inputs).transpose(1, 2).detach().cpu()  # raw, unchanged mapper output
        clean, sr = sf.read(path(row['clean_path']), dtype='float32', always_2d=True)
        target = torch.from_numpy(compute_mel(clean.mean(axis=1), sr=sr)).unsqueeze(0)
        if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
            raise RuntimeError(f'Non-finite cache for {uid}')
        cache[uid] = dict(predicted=predicted, target=target)
        torch.save(cache[uid], out/'cache'/f'{uid}.pt')
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(f'CACHE {index}/{len(rows)} {uid}', flush=True)
    return cache


@torch.no_grad()
def validate(step, rows, cache, postnet, generator, device, out, baseline=None):
    postnet.eval()
    generator.eval()
    items = []
    folder = out/'validation'/f'step_{step:08d}'
    folder.mkdir(parents=True, exist_ok=False)
    for row in rows:
        uid = row['utterance_id']
        predicted = cache[uid]['predicted'].to(device)
        target = cache[uid]['target'].to(device)
        refined = postnet.refine(predicted)
        if step == 0 and not torch.equal(refined, predicted):
            raise RuntimeError('Initial postnet must exactly preserve frozen-mapper output')
        mel_loss = float(mel_l1(refined, target))
        generated = generator(refined.clamp(MEL_CLAMP_MIN, MEL_CLAMP_MAX)).squeeze().cpu().numpy()
        if not np.isfinite(generated).all():
            raise RuntimeError(f'Non-finite generated audio: {uid}')
        sf.write(folder/f'{uid}.wav', np.clip(generated, -1, 1), 22050, subtype='PCM_16')
        reference = load_audio_16k(out/'references'/f'{uid}.wav')
        estimate = load_audio_16k(folder/f'{uid}.wav')
        items.append(dict(utterance_id=uid, mel_l1=mel_loss,
                          stoi=float(compute_stoi(reference, estimate)), pesq=float(compute_pesq(reference, estimate)),
                          residual_abs_mean=float((refined-predicted).abs().mean())))
    record = dict(step=step, items=items, **{k: float(np.mean([r[k] for r in items]))
                  for k in ('mel_l1', 'stoi', 'pesq', 'residual_abs_mean')})
    if not all(np.isfinite(r[k]) for r in items for k in ('mel_l1', 'stoi', 'pesq', 'residual_abs_mean')):
        raise RuntimeError('Non-finite validation result')
    if baseline is not None:
        record.update(baseline_step=baseline['step'], delta_stoi=record['stoi']-baseline['stoi'],
                      delta_pesq=record['pesq']-baseline['pesq'])
    append_record(out/'validation.jsonl', record)
    postnet.train()
    return record


@torch.no_grad()
def fixed_train_loss(postnet, rows, cache, device):
    postnet.eval()
    values = [float(mel_l1(postnet.refine(cache[r['utterance_id']]['predicted'].to(device)),
                          cache[r['utterance_id']]['target'].to(device))) for r in rows]
    postnet.train()
    return float(np.mean(values))


def save_checkpoint(out, step, postnet, optimizer, args, provenance, rng):
    dest = out/f'postnet_{step:08d}.pt'
    if dest.exists():
        raise FileExistsError(dest)
    temp = dest.with_suffix('.tmp')
    torch.save(dict(step=step, postnet=postnet.state_dict(), optimizer=optimizer.state_dict(),
                    postnet_config=postnet.config, args=vars(args), provenance=provenance,
                    torch_rng=torch.get_rng_state(), sampling_rng=rng.getstate()), temp)
    os.replace(temp, dest)
    print('CHECKPOINT '+str(dest), flush=True)


def train(args):
    os.chdir(ROOT)
    if args.max_steps < 1 or args.val_items < 1 or args.validation_interval < 1 or args.checkpoint_interval < 1:
        raise ValueError('Steps, validation/checkpoint intervals and val-items must be positive')
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.grad_clip) or args.grad_clip <= 0:
        raise ValueError('Positive finite lr and grad-clip required')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    resume_state = None
    resume_path = path(args.resume) if args.resume else None
    start = 0
    if resume_path is not None:
        resume_state = torch.load(resume_path, map_location='cpu', weights_only=True, mmap=True)
        required = {'step', 'postnet', 'optimizer', 'postnet_config', 'provenance', 'torch_rng', 'sampling_rng'}
        if not required.issubset(resume_state):
            raise ValueError(f'Incomplete postnet checkpoint: {resume_path}')
        start = int(resume_state['step'])
        if start < 1 or args.max_steps <= start:
            raise ValueError(f'--max-steps is the total target and must exceed resume step {start}')
        expected_config = dict(mel_channels=80, hidden_channels=args.hidden_channels, layers=5, kernel_size=5)
        if resume_state['postnet_config'] != expected_config:
            raise ValueError('Postnet architecture differs; use the checkpoint hidden-channels setting')
    out = path(args.output_dir) if args.output_dir else ROOT/'results/checkpoints/postnet'/datetime.now().strftime('run_%Y%m%d_%H%M%S')
    allowed = ROOT/'results/checkpoints/postnet'
    if out.resolve() == allowed.resolve() or not out.resolve().is_relative_to(allowed.resolve()):
        raise ValueError('Use a NEW subdirectory under results/checkpoints/postnet')
    out.mkdir(parents=True, exist_ok=False)
    split_rows = pairs(args.manifest, args.splits)
    val_rows = split_rows['val'][:args.val_items]
    if len(val_rows) != args.val_items:
        raise ValueError('Not enough validation rows')
    cache_rows = split_rows['train']+val_rows  # TEST audio is never read.
    mapper_path = path(args.mapper_checkpoint)
    vocoder_path = path('vendor/hifi-gan/checkpoints/UNIVERSAL_V1/g_02500000' if args.vocoder == 'universal'
                        else 'results/checkpoints/hifigan_thai/g_00010000')
    immutable_files = [mapper_path, vocoder_path, ROOT/'src/models/mapper.py', ROOT/'src/training/joint_finetune.py']
    file_hashes = {str(p): digest(p) for p in immutable_files}
    provenance = dict(files_sha256=file_hashes, trainer_sha256=digest(Path(__file__)),
                      postnet_source_sha256=digest(ROOT/'src/models/postnet.py'), manifest_sha256=digest(path(args.manifest)),
                      splits_sha256=digest(path(args.splits)), train_ids=[r['utterance_id'] for r in split_rows['train']],
                      val_ids=[r['utterance_id'] for r in val_rows], test_used=False,
                      alignment='Resize refined mel only in loss to clean length; inference uses distorted length',
                      predicted_mel='raw unchanged mapper output; clamp only before vocoder',
                      objective='L1 mel only; no adversarial or STFT loss', device=str(device))
    if resume_state is not None:
        previous = resume_state['provenance']
        for key in ('manifest_sha256', 'splits_sha256', 'postnet_source_sha256', 'alignment', 'train_ids'):
            if previous.get(key) != provenance[key]:
                raise ValueError(f'Resume input/protocol differs: {key}')
        # Compare hashes, not absolute paths, so a checkpoint can move between Mac and Colab.
        previous_hashes = set(previous['files_sha256'].values())
        if any(file_hashes[str(p)] not in previous_hashes for p in (mapper_path, vocoder_path)):
            raise ValueError('Resume requires the same frozen mapper and vocoder checkpoints')
        provenance['resume'] = dict(path=str(resume_path), sha256=digest(resume_path), step=start,
                                    optimizer_restored=True, lr_override=args.lr)
    (out/'run.json').write_text(json.dumps(dict(args=vars(args), provenance=provenance), indent=2)+'\n')
    encoder = Wav2Vec2ContentEncoder(device=device, layer=9)
    encoder.model.eval().requires_grad_(False)
    mapper = load_mapper(mapper_path, device, ROOT/'configs/model.yaml', ROOT/'configs/train.yaml', layer=9)
    mapper.eval().requires_grad_(False)
    before = dict(encoder=state_digest(encoder.model), mapper=state_digest(mapper))
    cache = build_cache(cache_rows, mapper, encoder, device, out)
    after = dict(encoder=state_digest(encoder.model), mapper=state_digest(mapper))
    assert before == after
    assert all(not p.requires_grad and p.grad is None for m in (encoder.model, mapper) for p in m.parameters())
    (out/'frozen_audit.json').write_text(json.dumps(dict(before=before, after_cache=after, unchanged=True), indent=2))
    del mapper, encoder
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    config = AttrDict(json.loads(vocoder_path.with_name('config.json').read_text()))
    assert (config.sampling_rate, config.num_mels) == (22050, 80)
    generator = Generator(config).to(device)
    generator.load_state_dict(torch.load(vocoder_path, map_location='cpu', weights_only=True)['generator'], strict=True)
    generator.eval()
    generator.remove_weight_norm()
    generator.requires_grad_(False)
    generator_hash = state_digest(generator)
    (out/'references').mkdir()
    for row in val_rows:
        audio, sr = sf.read(path(row['clean_path']), dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 22050:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        sf.write(out/'references'/f'{row["utterance_id"]}.wav', audio, 22050, subtype='PCM_16')
    postnet = Postnet(hidden_channels=args.hidden_channels).to(device)
    print('POSTNET parameters:', sum(p.numel() for p in postnet.parameters()), flush=True)
    optimizer = torch.optim.Adam(postnet.parameters(), lr=args.lr)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in postnet.parameters()}
    # Fixed train subset makes the L1 trend comparable despite utterance sampling.
    monitor_rows = split_rows['train'][::max(1, len(split_rows['train'])//16)][:16]
    initial_train = fixed_train_loss(postnet, monitor_rows, cache, device)
    append_record(out/'train_monitor.jsonl', dict(step=0, mel_l1=initial_train, utterance_ids=[r['utterance_id'] for r in monitor_rows]))
    baseline = validate(0, val_rows, cache, postnet, generator, device, out)
    validation = [baseline]
    if resume_state is not None:
        postnet.load_state_dict(resume_state['postnet'], strict=True)
        optimizer.load_state_dict(resume_state['optimizer'])
        # Adam moments/step counters continue, but the user's new LR takes precedence.
        for group in optimizer.param_groups:
            group['lr'] = args.lr
        torch.set_rng_state(resume_state['torch_rng'])
        rng.setstate(resume_state['sampling_rng'])
        del resume_state
        print(f'RESUME step={start} target={args.max_steps} lr={args.lr} checkpoint={resume_path}', flush=True)
        append_record(out/'train_monitor.jsonl', dict(step=start, mel_l1=fixed_train_loss(postnet, monitor_rows, cache, device)))
        validation.append(validate(start, val_rows, cache, postnet, generator, device, out, baseline))
    times, losses, gradients = [], [], []
    for step in range(start+1, args.max_steps+1):
        started = time.perf_counter()
        item = cache[rng.choice(split_rows['train'])['utterance_id']]
        predicted, target = item['predicted'].to(device), item['target'].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = mel_l1(postnet.refine(predicted), target)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(postnet.parameters(), args.grad_clip, error_if_nonfinite=True))
        if norm <= 0:
            raise RuntimeError('No postnet gradient')
        after_norm = float(torch.stack([p.grad.detach().norm() for p in postnet.parameters() if p.grad is not None]).norm())
        if not np.isfinite(after_norm) or after_norm > args.grad_clip*1.001:
            raise RuntimeError('Invalid clipped gradient')
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        elapsed = time.perf_counter()-started
        times.append(elapsed); losses.append(float(loss.detach())); gradients.append(norm)
        record = dict(step=step, mel_l1=losses[-1], grad_pre_clip=norm, grad_post_clip=after_norm, seconds=elapsed)
        with (out/'losses.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False)+'\n')
        if step == start+1 or step % 25 == 0:
            print('TRAIN '+json.dumps(record), flush=True)
        del loss
        if step % args.validation_interval == 0 or step == args.max_steps:
            append_record(out/'train_monitor.jsonl', dict(step=step, mel_l1=fixed_train_loss(postnet, monitor_rows, cache, device)))
            validation.append(validate(step, val_rows, cache, postnet, generator, device, out, baseline))
            assert state_digest(generator) == generator_hash
            assert all(not p.requires_grad and p.grad is None for p in generator.parameters())
        if step % args.checkpoint_interval == 0 or step == args.max_steps:
            save_checkpoint(out, step, postnet, optimizer, args, provenance, rng)
    assert {str(p): digest(p) for p in immutable_files} == file_hashes
    audit = json.loads((out/'frozen_audit.json').read_text())
    audit.update(vocoder_unchanged=state_digest(generator) == generator_hash, source_files_and_checkpoints_unchanged=True)
    (out/'frozen_audit.json').write_text(json.dumps(audit, indent=2))
    summary = dict(steps=args.max_steps-start, start_step=start, last_step=args.max_steps,
                   all_losses_finite=True, mean_step_seconds=float(np.mean(times)),
                   first_50_loss_mean=float(np.mean(losses[:50])), last_50_loss_mean=float(np.mean(losses[-50:])),
                   grad_pre_clip_min=min(gradients), grad_pre_clip_max=max(gradients), frozen_audit=audit,
                   validation_trend=[{k: v for k, v in r.items() if k != 'items'} for r in validation],
                   final_stoi_improved=validation[-1]['stoi'] > baseline['stoi'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('FINISHED '+json.dumps(summary), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--manifest', default='data/manifest_w5.csv')
    p.add_argument('--splits', default='data/splits_w5.json')
    p.add_argument('--mapper-checkpoint', default='results/checkpoints/mapper_layer9.pt')
    p.add_argument('--vocoder', choices=['universal', 'thai'], default='universal')
    p.add_argument('--max-steps', type=int, default=500, help='Total target step, including updates before --resume')
    p.add_argument('--resume', help='Restore postnet, Adam and RNG states into a NEW output directory; --lr overrides saved LR')
    p.add_argument('--hidden-channels', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-4, help='Adam LR; also overrides the restored optimizer LR when resuming')
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--validation-interval', type=int, default=100)
    p.add_argument('--checkpoint-interval', type=int, default=100)
    p.add_argument('--val-items', type=int, default=3)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--output-dir')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())

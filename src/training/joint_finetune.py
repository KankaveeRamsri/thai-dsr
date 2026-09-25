"""Joint mapper/HiFi-GAN fine-tuning. Existing models and datasets remain untouched."""
from __future__ import annotations
import argparse
import csv
import gc
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from src.inference.run import load_mapper, MEL_CLAMP_MIN, MEL_CLAMP_MAX
from src.models.encoder import Wav2Vec2ContentEncoder
from src.training.dataset import interpolate_embedding
from src.training.finetune_hifigan import (
    Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator,
    discriminator_loss, feature_loss, generator_loss, load_config,
    choose_device, set_requires_grad,
)
from src.utils.mel import compute_mel_tensor
from src.evaluation.metrics import compute_stoi, compute_pesq

ROOT = Path(__file__).resolve().parents[2]


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def default_mapper():
    # The latest controlled pipeline experiment is authoritative, not the older CLI default.
    p = ROOT / 'results/audio_compare/w6_vocoder_ft/provenance.json'
    if not p.is_file():
        raise FileNotFoundError('Pass --mapper-checkpoint; pipeline provenance is absent')
    metadata = json.loads(p.read_text())
    checkpoint = path(metadata['mapper'])
    if sha256(checkpoint) != metadata['mapper_sha256']:
        raise ValueError('Pipeline mapper differs from recorded provenance; pass --mapper-checkpoint')
    return checkpoint


def load_pairs(manifest, split_file):
    splits = json.loads(path(split_file).read_text())['splits']
    ids = list(itertools.chain.from_iterable(splits.values()))
    if len(ids) != len(set(ids)):
        raise ValueError('Split leakage or duplicate IDs')
    with path(manifest).open() as f:
        rows = [r for r in csv.DictReader(f) if r['severity'] == 'severe']
    by_id = {r['utterance_id']: r for r in rows}
    if len(rows) != len(by_id) or set(by_id) != set(ids):
        raise ValueError('Manifest and split IDs must match exactly, one severe pair per ID')
    for r in rows:
        for key in ('clean_path', 'distorted_path'):
            if not path(r[key]).is_file():
                raise FileNotFoundError(r[key])
    return {s: [by_id[i] for i in sorted(splits[s])] for s in ('train', 'val', 'test')}


def load_wave(p, sr=22050):
    x, original_sr = sf.read(path(p), dtype='float32', always_2d=True)
    x = torch.from_numpy(x.mean(axis=1)).unsqueeze(0)
    if original_sr != sr:
        x = AF.resample(x, original_sr, sr)
    return x


class FrozenContent:
    """Same frozen XLSR for input and content; generated branch retains autograd."""
    def __init__(self, device):
        self.device = device
        self.wrapper = Wav2Vec2ContentEncoder(device=device, layer=9)
        self.model = self.wrapper.model.eval()
        self.model.requires_grad_(False)
        # hidden_states[9] is captured BEFORE block 10; keep ten blocks so the
        # stable-layer-norm encoder's final normalization cannot change layer 9.
        probe = torch.linspace(-0.1, 0.1, 8000, device=device).unsqueeze(0)
        with torch.no_grad():
            expected = self.model(probe, output_hidden_states=True).hidden_states[9]
            self.model.encoder.layers = torch.nn.ModuleList(list(self.model.encoder.layers[:10]))
            actual = self.model(probe, output_hidden_states=True).hidden_states[9]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print('Verified exact layer-9 equivalence after omitting unused upper blocks.', flush=True)

    def encode_wave(self, audio, sr=22050):
        # CPU resampling is differentiable and avoids backend-specific sinc limitations.
        x = AF.resample(audio.cpu(), sr, 16000).to(self.device) if sr != 16000 else audio.to(self.device)
        x = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-7)
        return self.model(x, output_hidden_states=True).hidden_states[9]

    @torch.no_grad()
    def input_embedding(self, p):
        # Reuse exactly the existing feature extractor and input resampling path.
        return self.wrapper.encode(str(path(p)))


def loss_mel(wave):
    # Explicit CPU STFT keeps the complex transform portable; .cpu() preserves gradients.
    return compute_mel_tensor(wave.squeeze(1).cpu(), fmax=None)


def grad_norm(module):
    norms = [p.grad.detach().float().norm().cpu() for p in module.parameters() if p.grad is not None]
    value = torch.stack(norms).norm() if norms else torch.tensor(0.)
    if not torch.isfinite(value) or value <= 0:
        raise RuntimeError(f'Invalid/zero gradient in {type(module).__name__}: {value}')
    return float(value)


def gan_scale(step, warmup_steps, ramp_steps):
    """Global, 1-based update schedule; resume continues at its saved step."""
    if min(warmup_steps, ramp_steps) < 0:
        raise ValueError('GAN warmup/ramp steps must be nonnegative')
    if step <= warmup_steps:
        return 0.0
    return min(1.0, (step - warmup_steps) / ramp_steps) if ramp_steps else 1.0


def clip_optimizer_gradients(modules, max_norm):
    """Clip the combined optimizer norm, and measure it again after clipping."""
    if not np.isfinite(max_norm) or max_norm <= 0:
        raise ValueError('grad-clip must be finite and positive')
    parameters = [p for module in modules for p in module.parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError('Optimizer has no gradients to clip')
    before = float(torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True))
    after = float(torch.stack([p.grad.detach().float().norm() for p in parameters]).norm())
    if not np.isfinite(after) or after > max_norm * 1.001:
        raise RuntimeError(f'Gradient clipping failed: {after} > {max_norm}')
    return before, after


def weighted_loss_gradients(terms, predicted):
    """Compare loss gradients at the SAME mapper output, before optimizer clipping."""
    result = {}
    for name, term in terms.items():
        if not term.requires_grad:  # GAN branches are not evaluated during warmup.
            result[name] = 0.0
            continue
        gradient = torch.autograd.grad(term, predicted, retain_graph=True)[0]
        norm = float(gradient.detach().float().norm())
        if not np.isfinite(norm):
            raise RuntimeError(f'Non-finite {name} gradient at mapper output')
        result[name] = norm
    return result


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def save_joint(out, step, mapper, generator, mpd, msd, og, od, args, config, init, rng):
    destination = out / f'joint_{step:08d}.pt'
    if destination.exists():
        raise FileExistsError(destination)
    if shutil.disk_usage(out).free < 2 * 1024**3:
        raise OSError('Less than 2 GiB free; refusing checkpoint write')
    payload = dict(step=step, mapper=mapper.state_dict(), generator=generator.state_dict(),
                   mpd=mpd.state_dict(), msd=msd.state_dict(), optim_g=og.state_dict(), optim_d=od.state_dict(),
                   args=vars(args), vocoder_config=dict(config), initialization=init,
                   torch_rng=torch.get_rng_state(), sampling_rng=rng.getstate())
    temporary = destination.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    print(f'Saved joint checkpoint: {destination}', flush=True)


@torch.no_grad()
def validate(step, rows, get_item, mapper, generator, device, out, baseline=None):
    mapper.eval()
    generator.eval()
    scores = []
    for row in rows:
        embedding, clean, distorted_frames = get_item(row)
        inputs = interpolate_embedding(embedding, distorted_frames).unsqueeze(0).to(device)
        mel = mapper(inputs).clamp(MEL_CLAMP_MIN, MEL_CLAMP_MAX).transpose(1, 2)
        audio = generator(mel).squeeze().cpu().numpy()
        audio = np.clip(audio, -1, 1)
        folder = out / 'validation' / f'step_{step:08d}'
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / f'{row["utterance_id"]}.wav', audio, 22050, subtype='PCM_16')
        # Measure saved PCM audio, matching the previous evaluation protocol.
        saved, _ = sf.read(folder / f'{row["utterance_id"]}.wav', dtype='float32')
        reference = AF.resample(clean, 22050, 16000).squeeze().numpy()
        estimate = AF.resample(torch.from_numpy(saved), 22050, 16000).numpy()
        scores.append(dict(utterance_id=row['utterance_id'], stoi=float(compute_stoi(reference, estimate)),
                           pesq=float(compute_pesq(reference, estimate))))
    record = dict(step=step, items=scores, stoi=float(np.mean([s['stoi'] for s in scores])),
                  pesq=float(np.mean([s['pesq'] for s in scores])))
    if not all(np.isfinite(s[k]) for s in scores for k in ('stoi', 'pesq')):
        raise RuntimeError('Non-finite validation metric')
    if baseline is not None:
        record['baseline_step'] = baseline['step']
        record['delta_stoi'] = record['stoi'] - baseline['stoi']
        record['delta_pesq'] = record['pesq'] - baseline['pesq']
    with (out / 'validation.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    print('VALIDATION ' + json.dumps(record), flush=True)
    mapper.train()
    generator.train()
    return record


def train(args):
    os.chdir(ROOT)
    device = choose_device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    if args.segment_samples < 8192 or args.segment_samples % 256:
        raise ValueError('segment-samples must be >=8192 and divisible by 256')
    if min(args.adv_weight, args.fm_weight, args.mel_weight, args.content_weight) <= 0:
        raise ValueError('All four loss weights must be positive')
    if args.max_steps < 1 or args.val_items < 1:
        raise ValueError('Positive max-steps and val-items required')
    gan_scale(1, args.gan_warmup_steps, args.gan_ramp_steps)
    if not np.isfinite(args.grad_clip) or args.grad_clip <= 0:
        raise ValueError('grad-clip must be finite and positive')
    if min(args.validation_interval, args.checkpoint_interval, args.loss_grad_interval) < 0 or args.log_interval < 1:
        raise ValueError('Intervals must be nonnegative; log-interval must be positive')
    mapper_path = path(args.mapper_checkpoint) if args.mapper_checkpoint else default_mapper()
    pretrained = path('results/checkpoints/hifigan_thai' if args.vocoder_init == 'thai' else 'vendor/hifi-gan/checkpoints/UNIVERSAL_V1')
    gpath = pretrained / ('g_00010000' if args.vocoder_init == 'thai' else 'g_02500000')
    dpath = path(args.discriminator_checkpoint) if args.discriminator_checkpoint else pretrained / 'do_latest'
    config = load_config(pretrained)
    assert (config.sampling_rate, config.hop_size, config.num_mels) == (22050, 256, 80)
    pairs = load_pairs(args.manifest, args.splits)
    out = path(args.output_dir) if args.output_dir else ROOT / 'results/checkpoints/joint_finetune' / datetime.now().strftime('run_%Y%m%d_%H%M%S')
    allowed = ROOT / 'results/checkpoints/joint_finetune'
    if not out.resolve().is_relative_to(allowed.resolve()) or out.resolve() == allowed.resolve():
        raise ValueError('Use a NEW run subdirectory under results/checkpoints/joint_finetune')
    out.mkdir(parents=True, exist_ok=False)
    init = dict(mapper=str(mapper_path), mapper_sha256=sha256(mapper_path), generator=str(gpath),
                generator_sha256=sha256(gpath), discriminator=str(dpath) if dpath.is_file() else 'random',
                split_counts={k: len(v) for k, v in pairs.items()}, device=str(device),
                training_source_sha256=sha256(Path(__file__)))
    (out / 'run.json').write_text(json.dumps(dict(args=vars(args), initialization=init), indent=2))
    print(json.dumps(init, indent=2), flush=True)
    content = FrozenContent(torch.device(args.content_device))
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    mapper = load_mapper(mapper_path, device, ROOT/'configs/model.yaml', ROOT/'configs/train.yaml', layer=9).train()
    generator = Generator(config).to(device).train()
    generator.load_state_dict(torch.load(gpath, map_location='cpu', weights_only=True)['generator'])
    mpd, msd = MultiPeriodDiscriminator().to(device), MultiScaleDiscriminator().to(device)
    if dpath.is_file():
        state = torch.load(dpath, map_location='cpu', weights_only=True, mmap=True)
        if args.vocoder_init == 'thai' and not args.discriminator_checkpoint and state['steps'] != 10000:
            raise ValueError('do_latest does not match g_00010000')
        mpd.load_state_dict(state['mpd'])
        msd.load_state_dict(state['msd'])
        del state
    elif args.vocoder_init == 'thai' or args.discriminator_checkpoint:
        raise FileNotFoundError(dpath)
    else:
        print('UNIVERSAL_V1 has no discriminator checkpoint: initializing MPD/MSD randomly.', flush=True)
    og = torch.optim.AdamW([{'params': mapper.parameters(), 'lr': args.mapper_lr},
                           {'params': generator.parameters(), 'lr': args.generator_lr}],
                          betas=(config.adam_b1, config.adam_b2), weight_decay=0)
    od = torch.optim.AdamW(itertools.chain(mpd.parameters(), msd.parameters()), lr=args.discriminator_lr,
                          betas=(config.adam_b1, config.adam_b2), weight_decay=0)
    start = 0
    if args.resume:
        state = torch.load(path(args.resume), map_location='cpu', weights_only=True)
        for key, model in [('mapper', mapper), ('generator', generator), ('mpd', mpd), ('msd', msd)]:
            model.load_state_dict(state[key])
        og.load_state_dict(state['optim_g'])
        od.load_state_dict(state['optim_d'])
        start = int(state['step'])
        rng.setstate(state['sampling_rng'])
        torch.set_rng_state(state['torch_rng'])
        del state
        if start >= args.max_steps:
            raise ValueError('max-steps must exceed resume step')
    cache = {}
    def get_item(row):
        uid = row['utterance_id']
        if uid not in cache:
            emb = content.input_embedding(row['distorted_path'])
            clean = load_wave(row['clean_path'])
            distorted = load_wave(row['distorted_path'])
            cache[uid] = (emb, clean, distorted.shape[-1] // 256)
        return cache[uid]
    val_rows = pairs['val'][:args.val_items]
    baseline = validate(start, val_rows, get_item, mapper, generator, device, out)
    validations = [baseline]
    times = []
    gradient_checks = {}
    for step in range(start + 1, args.max_steps + 1):
        synchronize(device)
        started = time.perf_counter()
        row = rng.choice(pairs['train'])
        emb, clean, _ = get_item(row)
        frames = clean.shape[-1] // 256
        segment_frames = min(args.segment_samples // 256, frames)
        begin = rng.randint(0, frames - segment_frames)
        # Frozen embeddings retain whole-utterance context. Train the BiLSTM on
        # a bounded window with context on both sides, avoiding variable-length
        # MPS backward graphs and memory growth on 8 GiB machines.
        aligned = interpolate_embedding(emb, frames).T.unsqueeze(0)
        context_frames = 32
        aligned = F.pad(aligned, (context_frames, context_frames), mode='replicate')
        inputs = aligned[:, :, begin:begin+segment_frames+2*context_frames].transpose(1, 2).to(device)
        predicted = mapper(inputs).clamp(MEL_CLAMP_MIN, MEL_CLAMP_MAX)
        predicted = predicted[:, context_frames:context_frames+segment_frames].transpose(1, 2)
        generated = generator(predicted)
        real = clean[:, begin*256:(begin+segment_frames)*256].unsqueeze(1).to(device)
        assert generated.shape == real.shape
        for d in (mpd, msd):
            set_requires_grad(d, True)
            d.train()
        od.zero_grad(set_to_none=True)
        disc = 0
        for d in (mpd, msd):
            dr, df, _, _ = d(real, generated.detach())
            disc = disc + discriminator_loss(dr, df)[0]
        if not torch.isfinite(disc):
            raise RuntimeError('Non-finite discriminator loss')
        disc.backward()
        grad_norm(mpd), grad_norm(msd)  # Check that both discriminators receive gradients.
        dnorm, dnorm_after = clip_optimizer_gradients((mpd, msd), args.grad_clip)
        od.step()
        od.zero_grad(set_to_none=True)
        for d in (mpd, msd):
            set_requires_grad(d, False)
            d.eval()  # Do not update spectral-norm buffers during generator pass.
        og.zero_grad(set_to_none=True)
        scale = gan_scale(step, args.gan_warmup_steps, args.gan_ramp_steps)
        adv, fm = generated.new_zeros(()), generated.new_zeros(())
        if scale > 0:
            for d in (mpd, msd):
                _, df, fr, ff = d(real, generated)
                adv = adv + generator_loss(df)[0]
                fm = fm + feature_loss(fr, ff) / 2  # vendor already multiplies by two
            del fr, ff
        mel = F.l1_loss(loss_mel(generated), loss_mel(real)).to(device)
        with torch.no_grad():
            target_content = content.encode_wave(real.squeeze(1))
        generated_content = content.encode_wave(generated.squeeze(1))  # MUST keep autograd
        perceptual = F.l1_loss(generated_content, target_content).to(device)
        terms = dict(adversarial=scale*args.adv_weight*adv, feature_matching=scale*args.fm_weight*fm,
                     mel=args.mel_weight*mel, content=args.content_weight*perceptual)
        total = sum(terms.values())
        if not torch.isfinite(total):
            raise RuntimeError('Non-finite generator loss')
        for name, term in [('content', perceptual)] + ([('adversarial', adv)] if scale > 0 else []):
            if name not in gradient_checks:
                grad = torch.autograd.grad(term, predicted, retain_graph=True)[0]
                value = float(grad.norm())
                if not np.isfinite(value) or value <= 0:
                    raise RuntimeError(f'{name} has no finite gradient to mapper output')
                gradient_checks[name] = dict(step=step, to_mapper_output_grad_norm=value)
                del grad
                (out / 'gradient_checks.json').write_text(json.dumps(gradient_checks, indent=2))
                print('GRADIENT_CHECK ' + json.dumps(gradient_checks), flush=True)
        loss_gradients = None
        if (step in {start+1, args.gan_warmup_steps+1, args.gan_warmup_steps+args.gan_ramp_steps}
                or (args.loss_grad_interval and step % args.loss_grad_interval == 0)):
            loss_gradients = weighted_loss_gradients(terms, predicted)
        total.backward()
        mnorm, gnorm = grad_norm(mapper), grad_norm(generator)
        if any(p.grad is not None or p.requires_grad for p in content.model.parameters()):
            raise RuntimeError('Frozen wav2vec2 received parameter gradients')
        gnorm_combined, gnorm_after = clip_optimizer_gradients((mapper, generator), args.grad_clip)
        mnorm_after, generator_norm_after = grad_norm(mapper), grad_norm(generator)
        og.step()
        synchronize(device)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        record = dict(step=step, utterance_id=row['utterance_id'], total=float(total.detach()), discriminator=float(disc.detach()),
                      adversarial=float(adv.detach()), feature_matching_raw=float(fm.detach()), mel=float(mel.detach()), content=float(perceptual.detach()),
                      mapper_grad=mnorm, generator_grad=gnorm, discriminator_grad=dnorm, seconds=elapsed,
                      mapper_grad_post_clip=mnorm_after, generator_grad_post_clip=generator_norm_after,
                      generator_optimizer_grad_pre_clip=gnorm_combined, generator_optimizer_grad_post_clip=gnorm_after,
                      discriminator_grad_post_clip=dnorm_after, grad_clip=args.grad_clip,
                      gan_scale=scale, adversarial_weight=scale*args.adv_weight, feature_matching_weight=scale*args.fm_weight,
                      gan_losses_evaluated=scale > 0, weighted_losses={k: float(v.detach()) for k, v in terms.items()})
        if loss_gradients is not None:
            record['weighted_loss_grad_at_mapper_output'] = loss_gradients
        with (out / 'losses.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        if step == start+1 or step % args.log_interval == 0:
            print('TRAIN ' + json.dumps(record), flush=True)
        # Drop graphs before validation/checkpoint serialization.
        del generated, real, generated_content, target_content, total, terms, term, adv, fm, mel, perceptual, disc, predicted, dr, df, inputs
        og.zero_grad(set_to_none=True)
        if device.type == "mps":
            torch.mps.empty_cache()
        if step == args.max_steps or (args.validation_interval and step % args.validation_interval == 0):
            validations.append(validate(step, val_rows, get_item, mapper, generator, device, out, baseline))
        if step == args.max_steps or (args.checkpoint_interval and step % args.checkpoint_interval == 0):
            save_joint(out, step, mapper, generator, mpd, msd, og, od, args, config, init, rng)
    summary = dict(steps=args.max_steps-start, mean_seconds=float(np.mean(times)),
                   median_seconds=float(np.median(times)), all_losses_finite=True, output=str(out),
                   validation_trend=[{k: v for k, v in r.items() if k != 'items'} for r in validations])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('FINISHED ' + json.dumps(summary), flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['mps','cpu','cuda','auto'], default='mps')
    p.add_argument('--content-device', choices=['cpu', 'mps', 'cuda'], default='cpu', help='CPU conserves unified GPU memory; waveform gradients still flow')
    p.add_argument('--manifest', default='data/manifest_w5.csv')
    p.add_argument('--splits', default='data/splits_w5.json')
    p.add_argument('--mapper-checkpoint')
    p.add_argument('--vocoder-init', choices=['thai','universal'], default='thai')
    p.add_argument('--discriminator-checkpoint')
    p.add_argument('--output-dir')
    p.add_argument('--resume', help='Joint checkpoint to resume into a NEW output directory')
    p.add_argument('--max-steps', type=int, default=100)
    p.add_argument('--segment-samples', type=int, default=16384)
    p.add_argument('--mapper-lr', type=float, default=1e-5)
    p.add_argument('--generator-lr', type=float, default=1e-5)
    p.add_argument('--discriminator-lr', type=float, default=1e-5)
    p.add_argument('--adv-weight', type=float, default=1)
    p.add_argument('--fm-weight', type=float, default=2)
    p.add_argument('--mel-weight', type=float, default=45)
    p.add_argument('--content-weight', type=float, default=0.5, help='Conservative content anchor; inspect weighted-loss gradient diagnostics when tuning')
    p.add_argument('--grad-clip', type=float, default=1.0, help='Combined max L2 norm per optimizer, before its step')
    p.add_argument('--gan-warmup-steps', type=int, default=1000, help='Reconstruction-only generator updates; discriminators still train')
    p.add_argument('--gan-ramp-steps', type=int, default=1000, help='Linear GAN/FM ramp after warmup (0 means immediate full weight)')
    p.add_argument('--loss-grad-interval', type=int, default=1000, help='Log each weighted loss gradient at mapper output; 0 disables periodic checks')
    p.add_argument('--checkpoint-interval', type=int, default=500)
    p.add_argument('--validation-interval', type=int, default=1000)
    p.add_argument('--val-items', type=int, default=3)
    p.add_argument('--log-interval', type=int, default=10)
    p.add_argument('--seed', type=int, default=1234)
    return p.parse_args(argv)


if __name__ == '__main__':
    train(parse_args())

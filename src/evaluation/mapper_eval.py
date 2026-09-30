"""Shared mapper evaluation under the W5 smoke-validation protocol.

mel L1: masked L1 in batches of 4 with clean-length alignment.
Audio: the frozen UNIVERSAL_V1 HiFi-GAN at distorted-length alignment, scored
with STOI/PESQ against the clean reference.

``data(row)`` must return ``(embedding (T_emb, 1024), clean_mel (T_mel, 80),
distorted_mel_frame_count)`` for a manifest row.
"""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from src.evaluation.metrics import compute_pesq, compute_stoi, load_audio_16k
from src.inference.run import MEL_CLAMP_MAX, MEL_CLAMP_MIN, AttrDict, Generator
from src.training.train import masked_l1_loss

ROOT = Path(__file__).resolve().parents[2]
VOC = ROOT / 'vendor/hifi-gan/checkpoints/UNIVERSAL_V1/g_02500000'
W5_MANIFEST = ROOT / 'data/manifest_w5.csv'
W5_SPLITS = ROOT / 'data/splits_w5.json'


def interpolate(h, n):
    return F.interpolate(h.T[None], size=int(n), mode='linear', align_corners=False)[0].T


def rows_for(manifest, split_path):
    """Severe manifest rows per split, in manifest row order."""
    with open(manifest, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['severity'] == 'severe']
    splits = json.loads(Path(split_path).read_text(encoding='utf-8'))['splits']
    return {s: [r for r in rows if r['utterance_id'] in set(ids)] for s, ids in splits.items()}


def w5_eval_rows():
    """The 32 W5 val rows and the 8 fixed STOI/PESQ reference rows."""
    val = rows_for(W5_MANIFEST, W5_SPLITS)['val']
    return val, [val[i] for i in np.linspace(0, len(val) - 1, 8, dtype=int)]


def load_vocoder(device='cpu'):
    voc = Generator(AttrDict(json.loads(VOC.with_name('config.json').read_text())))
    voc.load_state_dict(torch.load(VOC, map_location='cpu', weights_only=True)['generator'])
    voc.remove_weight_norm()
    return voc.to(device).eval().requires_grad_(False)


@torch.no_grad()
def mel_l1(model, rows, data):
    device = next(model.parameters()).device
    total = 0.
    for start in range(0, len(rows), 4):
        rr = rows[start:start + 4]
        items = [data(r) for r in rr]
        lengths = torch.tensor([len(t) for _, t, _ in items])
        hs = pad_sequence([interpolate(h, n) for (h, _, _), n in zip(items, lengths)], batch_first=True)
        target = pad_sequence([t for _, t, _ in items], batch_first=True)
        hs, target, lengths = hs.to(device), target.to(device), lengths.to(device)
        total += float(masked_l1_loss(model(hs, lengths), target, lengths)) * len(rr)
    return total / len(rows)


@torch.no_grad()
def audio_metrics(model, voc, rows, data, folder):
    folder.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    items = []
    for row in rows:
        uid = row['utterance_id']
        h, _, frames = data(row)
        pred = model(interpolate(h, frames)[None].to(device), torch.tensor([frames], device=device)).cpu()
        wav = voc(pred.transpose(1, 2).clamp(MEL_CLAMP_MIN, MEL_CLAMP_MAX)).squeeze().cpu().numpy()
        assert np.isfinite(wav).all()
        dest = folder / (uid + '.wav')
        sf.write(dest, np.clip(wav, -1, 1), 22050, subtype='PCM_16')
        ref, est = load_audio_16k(ROOT / row['clean_path']), load_audio_16k(dest)
        item = dict(utterance_id=uid, stoi=float(compute_stoi(ref, est)))
        try:
            item['pesq'] = float(compute_pesq(ref, est))
        except Exception as exc:
            item.update(pesq=None, pesq_error=str(exc))
        items.append(item)
    pesqs = [v['pesq'] for v in items if v['pesq'] is not None]
    return dict(stoi=float(np.mean([v['stoi'] for v in items])),
                pesq=float(np.mean(pesqs)) if pesqs else None, pesq_valid=len(pesqs), items=items)

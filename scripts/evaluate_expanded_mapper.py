#!/usr/bin/env python3
"""Compare mapper checkpoints with the W5 smoke-validation protocol.

Reproduces the validation in src/training/train_mapper_mel_adversarial.py:
masked mel L1 over the 32 W5 val utterances (batches of 4, clean-length
alignment) and STOI/PESQ on the same 8 linspace-selected val utterances,
vocoded by the frozen UNIVERSAL_V1 HiFi-GAN at distorted-length alignment.
Also reports mel L1 over the whole expanded val split and STOI/PESQ on extra
new-val utterances. The test split is never touched.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.mapper_eval import (  # noqa: E402
    audio_metrics, load_vocoder, mel_l1, rows_for, w5_eval_rows,
)
from src.inference.run import compute_mel_frame_count  # noqa: E402
from src.models.mapper import build_mapper_from_config  # noqa: E402
from src.utils.mel import compute_mel  # noqa: E402

EMB = ROOT / 'data/embeddings/distorted'


class Data:
    def __init__(self, fp16):
        self.fp16, self.cache = fp16, {}

    def __call__(self, row):
        uid = row['utterance_id']
        if uid not in self.cache:
            h = np.load(EMB / f'{uid}_severe_layer09.npy')
            if self.fp16:
                h = h.astype(np.float16)
            clean, sr = sf.read(ROOT / row['clean_path'], dtype='float32', always_2d=True)
            self.cache[uid] = (torch.from_numpy(h.astype(np.float32)),
                               torch.from_numpy(compute_mel(clean.mean(axis=1), sr=sr).T.copy()),
                               compute_mel_frame_count(ROOT / row['distorted_path']))
        return self.cache[uid]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', action='append', required=True, help='name=path (repeatable)')
    p.add_argument('--expanded-manifest', default='data/manifest_expanded.csv')
    p.add_argument('--expanded-splits', default='data/splits_expanded.json')
    p.add_argument('--extra-audio', type=int, default=24, help='extra new-val utterances for STOI/PESQ')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    w5_val, audio_val = w5_eval_rows()
    exp_val = rows_for(ROOT / args.expanded_manifest, ROOT / args.expanded_splits)['val']
    w5_ids = {r['utterance_id'] for r in w5_val}
    new_val = [r for r in exp_val if r['utterance_id'] not in w5_ids]
    extra = [new_val[i] for i in np.linspace(0, len(new_val) - 1, args.extra_audio, dtype=int)] if args.extra_audio else []
    assert w5_ids <= {r['utterance_id'] for r in exp_val}, 'W5 val utterances left the val split'

    voc = load_vocoder()

    results = dict(audio_val_ids=[r['utterance_id'] for r in audio_val],
                   extra_audio_ids=[r['utterance_id'] for r in extra],
                   counts=dict(w5_val=len(w5_val), expanded_val=len(exp_val), new_val=len(new_val)),
                   models={})
    for spec in args.checkpoint:
        name, path = spec.split('=', 1)
        ckpt = torch.load(ROOT / path, map_location='cpu', weights_only=True)
        model = build_mapper_from_config(ckpt['model_config'])
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        for fp16 in (False, True) if name == 'baseline' else (False,):
            label = name + ('_fp16_embeddings' if fp16 else '')
            data = Data(fp16)
            entry = dict(checkpoint=path, epoch=ckpt.get('epoch'),
                         w5_val_mel_l1=mel_l1(model, w5_val, data),
                         expanded_val_mel_l1=mel_l1(model, exp_val, data),
                         new_val_mel_l1=mel_l1(model, new_val, data) if new_val else None,
                         w5_audio=audio_metrics(model, voc, audio_val, data, out / label / 'w5_audio'))
            if extra:
                entry['extra_audio'] = audio_metrics(model, voc, extra, data, out / label / 'extra_audio')
            results['models'][label] = entry
            print(label, json.dumps({k: v for k, v in entry.items() if not isinstance(v, dict)}),
                  'stoi', entry['w5_audio']['stoi'], 'pesq', entry['w5_audio']['pesq'],
                  *(('extra_stoi', entry['extra_audio']['stoi'], 'extra_pesq', entry['extra_audio']['pesq']) if extra else ()),
                  flush=True)
    (out / 'comparison.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print('WROTE', out / 'comparison.json')


if __name__ == '__main__':
    main()

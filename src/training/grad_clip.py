"""Gradient clipping that rescales (never raises) and logs each call.

Shared by the mel-adversarial trainer and train.py's partial wav2vec2 unfreezing.
"""
import json

import torch


def clip(parameters, *, diagnostics=None, step=None, component=None, weight=None, max_norm=1.0):
    """Rescale finite gradients; log before an optimizer update or nonfinite failure.

    Float32 norm reductions can differ slightly before and after rescaling.
    A tiny post-clip overshoot is diagnostic, not a reason to abort training.
    """
    parameters = list(parameters)
    try:
        raw = float(torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True))
    except RuntimeError:
        raw = float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters if p.grad is not None])))
        entry = dict(step=step, component=component, adversarial_weight=weight,
                     raw_grad_norm=str(raw), error='gradient clipping failed')
        if diagnostics is not None:
            with diagnostics.open('a') as f:
                f.write(json.dumps(entry)+'\n')
        print('GRADIENT_ERROR', json.dumps(entry), flush=True)
        raise
    post = float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters if p.grad is not None])))
    entry = dict(step=step, component=component, adversarial_weight=weight,
                 raw_grad_norm=raw, post_clip_norm=post, old_bound_would_fail=post > max_norm * 1.00001)
    if diagnostics is not None:
        with diagnostics.open('a') as f:
            f.write(json.dumps(entry, allow_nan=False)+'\n')
    if post > max_norm * 1.00001:
        print('CLIP_ROUNDOFF', json.dumps(entry), flush=True)
    return raw, post

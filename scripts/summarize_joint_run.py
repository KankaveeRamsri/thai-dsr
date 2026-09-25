"""Summarize real joint-training logs; never substitutes metrics for missing records."""
import argparse
import json
import math
from pathlib import Path
import statistics

FIELDS = ('mapper_grad', 'generator_grad', 'generator_optimizer_grad_pre_clip',
          'mapper_grad_post_clip', 'generator_grad_post_clip',
          'generator_optimizer_grad_post_clip', 'discriminator_grad', 'discriminator_grad_post_clip')


def summarize(folder, expected_steps=None):
    folder = Path(folder)
    losses = [json.loads(line) for line in (folder/'losses.jsonl').read_text().splitlines()]
    validation = [json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
    if not losses or not validation:
        raise ValueError('Actual training and validation records are required')
    args = json.loads((folder/'run.json').read_text())['args']
    ranges = {k: dict(min=min(r[k] for r in losses), median=statistics.median(r[k] for r in losses),
                       max=max(r[k] for r in losses)) for k in FIELDS}
    numeric = [v for r in losses for v in r.values() if isinstance(v, (float, int))]
    all_finite = all(math.isfinite(v) for v in numeric)
    clip_ok = all(r[k] <= args['grad_clip']*1.001 for r in losses
                  for k in ('generator_optimizer_grad_post_clip', 'discriminator_grad_post_clip'))
    target = expected_steps if expected_steps is not None else args['max_steps']
    completed = losses[-1]['step'] == target and (folder/'summary.json').is_file()
    trend = [{k: v for k, v in row.items() if k != 'items'} for row in validation]
    report = dict(output=str(folder), completed=completed, target_step=target,
                  observed_steps=len(losses), last_step=losses[-1]['step'], all_logged_numbers_finite=all_finite,
                  measured_clip_bound_satisfied=clip_ok, gradient_norms=ranges,
                  validation_trend=trend, baseline_validation=trend[0],
                  reconstruction_only_steps=sum(r['gan_scale'] == 0 for r in losses),
                  ramp_steps=sum(0 < r['gan_scale'] < 1 for r in losses),
                  full_weight_steps=sum(r['gan_scale'] == 1 for r in losses),
                  mean_training_seconds=statistics.mean(r['seconds'] for r in losses),
                  weighted_loss_gradients=[dict(step=r['step'], **r['weighted_loss_grad_at_mapper_output'])
                                          for r in losses if 'weighted_loss_grad_at_mapper_output' in r])
    print(f'Observed {len(losses)} updates; last step {losses[-1]["step"]}/{target}; completed={completed}')
    print(f'All logged scalars finite: {all_finite}; measured clip bound: {clip_ok}')
    print(f'{"Gradient norm":<40} {"min":>12} {"median":>12} {"max":>12}')
    for key, values in ranges.items():
        print(f'{key:<40} {values["min"]:12.6f} {values["median"]:12.6f} {values["max"]:12.6f}')
    print('\nStep      VAL STOI      VAL PESQ       delta STOI      delta PESQ')
    for row in trend:
        print(f'{row["step"]:4d} {row["stoi"]:13.6f} {row["pesq"]:13.6f}'
              f' {row["stoi"]-trend[0]["stoi"]:+16.6f} {row["pesq"]-trend[0]["pesq"]:+15.6f}')
    print('This validation set is not the 8-utterance TEST set; 0.180 is not its baseline.')
    print('A post-clip norm <= 1 is a clipping check, not proof of improved intelligibility.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--expected-steps', type=int)
    parser.add_argument('--output', type=Path, help='Optional report JSON path')
    args = parser.parse_args()
    report = summarize(args.run_dir, args.expected_steps)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')

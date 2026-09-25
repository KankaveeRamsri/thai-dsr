# Joint fine-tuning stability investigation

## Findings before changing the trainer

The original generator-side objective was `1*adversarial + 2*feature_matching_raw + 45*mel + 2*content`. Vendor feature matching already multiplies by two; dividing it by two before applying the configured weight avoids a duplicate factor. Both optimizers were AdamW, learning rates 1e-5, betas (0.8, 0.99), weight decay zero. The mapper and generator share one optimizer; MPD and MSD share another.

Both optimizers ALREADY used `clip_grad_norm_` immediately before stepping, with a combined norm ceiling of 100. The logged mapper/generator norms were measured before clipping. The old discriminator log was the sum of MPD and MSD norms, not the combined L2 norm. The revision logs the true combined discriminator norm. Raw norms across millions of parameters have no universal healthy range of single digits. A clipped norm does not bound AdamW parameter updates in the same way as plain SGD, so clipping alone is not evidence of stable learning.

There was NO GAN warmup or ramp. Adversarial and feature matching were at full weight from the first update. Validation already evaluated three held-out VAL utterances at step 0 and periodically; the old CLI default interval was 50, while the Colab launcher passed 250. This revision makes the production default 1000 and records deltas against that run's initial validation.

User-reported TEST STOI (0.180 -> 0.146 at 2000 -> 0.139 at 10000) is evidence of degraded reconstruction, but does not by itself identify exploding gradients or prove the concept is sound. The Colab logs/checkpoints from that experiment are not locally available. Existing local hybrid logs contain only 35 updates, with raw mapper norms 100.26–359.87 and generator norms 124.08–4943.70; they do not show a completed 100-step smoke.

A real CPU integration probe of three steps, with content weight still 2, showed a raw generator norm of 1502.72 even with GAN/FM disabled on update 1. At the same mapper-output tensor, weighted mel/content gradient norms were 20.08/40.57 on step 1, 2.91/11.77 on step 2, and 6.69/18.65 on step 3. On the active GAN steps, adversarial norms were 0.90 and 0.06. This small diagnostic does not establish global loss dominance, but it refutes the claim that high raw norms here require adversarial gradients. It motivates reducing the content contribution for the controlled smoke.

## Changes and experiment

- Combined max L2 gradient norm **1.0** for each optimizer (`--grad-clip`), before `optimizer.step()`. Both before/after norms are measured and logged; non-finite gradients or violated clipping bounds fail the run.
- `--gan-warmup-steps 1000`: no GAN or feature-matching forward graph into the generator for the first 1000 updates. Discriminators still train on detached generated audio.
- `--gan-ramp-steps 1000`: on steps 1001–2000 the shared multiplier is `(step-1000)/1000`; it remains 1 after that. Schedule uses absolute global steps when resuming. Zero ramp steps selects immediate full weight after warmup.
- Full target weights: **adv 1, FM 2, mel 45, content 0.5**. The 4x content reduction is a conservative response to the measured diagnostic above, not a claim of optimal weights. Learning rates remain 1e-5 to avoid adding another simultaneous change.
- GAN/content connectivity checks occur on each branch's first active update. `weighted_loss_grad_at_mapper_output` periodically measures gradients from each weighted loss at the same predicted-mel tensor, without accumulating parameter gradients. During warmup, GAN losses are not evaluated; their logged zeros represent disabled branches, marked by `gan_losses_evaluated=false`.
- Default validation every 1000 steps on the same three VAL utterances; always evaluate at startup and at the final step too. Save per-utterance metrics/audio and deltas from the run's startup baseline in `validation.jsonl`.
- Default `--max-steps 100` is still a short connectivity run and does NOT cover the production warmup. Use the explicit 500-step smoke settings below to cover every phase.

The smoke is **500 real updates**, fresh original mapper + Approach-A vocoder/discriminators, seed 1234, original 16384-sample crop, warmup 100, ramp 200, full weight for steps 300–500, content 0.5, clipping 1. Validation and weighted-loss gradient diagnostics every 100 updates; checkpoints at 250 and 500. It does not resume the old degraded model. All old checkpoints are retained.

CPU was selected because CUDA and MPS are unavailable to Python in this session, and the UI tool denied Chrome access. A real three-step CPU probe averaged 7.44 seconds/update, making a local smoke feasible. This verifies CPU behavior; CUDA numerical results need a separate run. The self-contained `scripts/colab_stable_joint_smoke.ipynb` embeds the revised trainer under a new source-hash-based module name and saves a new Drive run directory, so it can repeat the experiment without changing the old Colab trainer or requiring a Git push.

```sh
# Choose a NEW directory for every experiment. CUDA command shown.
python -m src.training.joint_finetune \
  --device cuda --content-device cuda \
  --mapper-checkpoint results/checkpoints/mapper_layer9.pt \
  --max-steps 500 --gan-warmup-steps 100 --gan-ramp-steps 200 \
  --grad-clip 1 --content-weight 0.5 --validation-interval 100 --val-items 3 \
  --loss-grad-interval 100 --checkpoint-interval 250 \
  --output-dir results/checkpoints/joint_finetune/NEW_stable_smoke
python scripts/summarize_joint_run.py results/checkpoints/joint_finetune/NEW_stable_smoke --expected-steps 500
```

For a longer fresh experiment, production defaults are 1000 warmup + 1000 ramp. Do not resume an old step-10000 snapshot to test a fresh warmup: its global step has already passed it, and AdamW optimizer states are restored on resume. Three utterances and one seed are screening evidence, not a reliable significance test. No longer run is launched automatically by the new smoke notebook.

## Interpretation constraints

VAL step-0 STOI is approximately **0.28106**, PESQ **1.10700**, for these three utterances with original mapper + Approach-A vocoder. The TEST number **0.180** is from a different subset/model combination and cannot be a validation pass threshold. Compare VAL against its own startup baseline, then run the three-way real TEST evaluation once after training (`scripts/evaluate_joint_checkpoint.py`) using the exact same selection and metric utilities as the prior evaluation.

The training code still interpolates distorted embeddings to CLEAN duration; inference uses DISTORTED duration. Crop correspondence is approximate and does not compensate local tempo changes. Warmup and clipping do not fix this mismatch, content-loss misalignment, or metric/objective mismatch. If clipping and scheduling work but VAL/TEST regress, do not describe the run as a successful stabilization of intelligibility or automatically launch a longer run.

References: [PyTorch clip_grad_norm_](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html), [AdamW update definition](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).

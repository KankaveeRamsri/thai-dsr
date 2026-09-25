# Mel-only adversarial mapper: completed 300-step smoke summary

Recommendation: **do not prioritize an unchanged 2,000–3,000-step run as the next quality-improvement experiment.** This smoke passed numerical stability checks but did not improve audio metrics. The GAN is still evolving, so 300 steps do not establish that the approach can never help; however, the late trend shows the discriminator gaining ground without a corresponding STOI/PESQ benefit. More steps would be an exploratory convergence experiment, not an extension supported by positive quality evidence. No new training was started for this report.

## Run and evaluation scope

Run: `results/checkpoints/mel_adversarial_mapper/smoke300_20260925_224308_80333`. Completed all 300 updates. The summary reports all finite values, unchanged immutable files, frozen encoder/vocoder, and no test-set use. Independently checked all 300 consecutive training records and 600 gradient diagnostic records. The original launch used nohup + os.setsid() + disown.

The trained layer-9 mapper initializes this run exactly; Adam states and the small mel discriminator start fresh. Batch size 4, seed 42, CPU with four threads. Mapper LR 1e-5, discriminator LR 1e-4. Objective: `10 × masked mel-L1 + λ × generator adversarial loss`. λ is zero through step 50, ramps from step 51, reaches 0.1 at step 150, and stays there. Both optimizers use global gradient clipping at 1.0. The discriminator learns even during mapper warmup. The universal HiFi-GAN vocoder is frozen and used only for validation.

Mel-L1 covers all 32 validation utterances; STOI/PESQ cover the same fixed eight used for weighted-layer and spectral-aux experiments. All four audio evaluations have 8/8 valid PESQ results. Training/L1 uses clean-target duration; audio inference uses distorted duration. Existing local timing mismatch remains. No listening evaluation or statistical significance claim is made.

## Complete validation trend

Every distinct logged validation step is included, combining `validation.jsonl` and `epochs.jsonl`. Step 300 appears in both with identical shared metrics and is shown once. A dash means audio metrics were not evaluated at that epoch boundary. Clip rate and maximum raw gradient norm are cumulative; step 0 has no updates and therefore logs zeros.

| Step | val_mel_l1 | STOI | PESQ | discriminator_loss | generator_adversarial_loss | adversarial_weight | mapper_clip_rate | mapper_max_grad_norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.288527 | 0.261533 | 1.094355 | 0.448338 | 0.890800 | 0.000000 | 0.000000 | 0.000000 |
| 37 | 1.277187 | — | — | 0.268750 | 0.334485 | 0.000000 | 1.000000 | 26.547384 |
| 74 | 1.275546 | — | — | 0.252771 | 0.280237 | 0.024000 | 1.000000 | 26.547384 |
| 100 | 1.277939 | 0.262045 | 1.085167 | 0.247931 | 0.277796 | 0.050000 | 1.000000 | 26.547384 |
| 111 | 1.276644 | — | — | 0.245671 | 0.281126 | 0.061000 | 1.000000 | 28.872759 |
| 148 | 1.275163 | — | — | 0.236471 | 0.284282 | 0.098000 | 1.000000 | 33.382175 |
| 185 | 1.276334 | — | — | 0.221226 | 0.292376 | 0.100000 | 1.000000 | 33.382175 |
| 200 | 1.277653 | 0.259670 | 1.090617 | 0.212667 | 0.309885 | 0.100000 | 1.000000 | 33.382175 |
| 222 | 1.277908 | — | — | 0.197722 | 0.292600 | 0.100000 | 1.000000 | 33.382175 |
| 259 | 1.276036 | — | — | 0.176680 | 0.456639 | 0.100000 | 1.000000 | 33.382175 |
| 296 | 1.276931 | — | — | 0.152108 | 0.485398 | 0.100000 | 1.000000 | 33.382175 |
| 300 | 1.275986 | 0.258440 | 1.087187 | 0.156920 | 0.546468 | 0.100000 | 1.000000 | 33.382175 |

Step 0 is the relevant trained layer-9 baseline. Final mel-L1 improves by 0.012541 (0.973%), but STOI changes by -0.003093 and PESQ by -0.007168. STOI peaks at step 100 (only +0.000513 above baseline), then falls at both later evaluations. Every trained audio checkpoint has lower PESQ than baseline. Best logged mel-L1 is **1.275163 at step 148**, the `best.pt` epoch checkpoint; no audio metrics were measured at that step. `final.pt` and `step_000300.pt` represent step 300.

## Step-300 per-utterance audio metrics

Deltas use this run's step-0 scores, not another experiment's slightly different numerical baseline.

| Utterance | Baseline STOI | Step-300 STOI | Δ STOI | Baseline PESQ | Step-300 PESQ | Δ PESQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 002_ขอน้ำหนึ่งแก้วได้ไหม | 0.276471 | 0.279581 | +0.003110 | 1.107273 | 1.102904 | -0.004369 |
| cv_0014 | 0.199793 | 0.204298 | +0.004505 | 1.097741 | 1.103883 | +0.006141 |
| cv_0039 | 0.275261 | 0.264376 | -0.010885 | 1.057378 | 1.056558 | -0.000820 |
| cv_0076 | 0.200647 | 0.176601 | -0.024046 | 1.115440 | 1.106268 | -0.009171 |
| cv_0082 | 0.340329 | 0.340178 | -0.000151 | 1.075074 | 1.055287 | -0.019787 |
| cv_0127 | 0.284313 | 0.290529 | +0.006216 | 1.072118 | 1.065959 | -0.006159 |
| cv_0165 | 0.151464 | 0.152050 | +0.000587 | 1.078406 | 1.076464 | -0.001942 |
| cv_0190 | 0.363983 | 0.359904 | -0.004079 | 1.151410 | 1.130174 | -0.021235 |

STOI improves on 4/8 utterances and regresses on 4/8; PESQ improves on only 1/8 (`cv_0014`). The largest STOI regression is `cv_0076`. Small gains on some utterances do not offset the mean decline.

## Comparison with previous attempts

Final checkpoints are compared below, rather than selecting each run's best metric independently. The first four rows share eight audio IDs, 32-item validation L1 and the frozen universal vocoder. Postnet uses three different validation items (only one overlaps the eight) and its own L1 alignment convention. Joint fine-tuning uses those three items but an Approach-A vocoder that is then trained. Their absolute scores are historical context, not a common leaderboard. “Δ” is relative to each experiment's trained-model reference baseline; for the freshly initialized weighted/spectral models it is not relative to their random step-0 outputs. GAN losses are specific to this discriminator/objective and cannot be compared to non-GAN or waveform-GAN losses as equivalent metrics.

| Experiment / reference | Final step | Audio / L1 VAL count | VAL mel-L1 | STOI | PESQ | Δ STOI vs own trained baseline | Δ PESQ vs own trained baseline |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Layer-9 baseline (mel-GAN step 0) | 0 | 8 / 32 | 1.288527 | 0.261533 | 1.094355 | +0.000000 | +0.000000 |
| Mel-only adversarial | 300 | 8 / 32 | 1.275986 | 0.258440 | 1.087187 | -0.003093 | -0.007168 |
| Weighted-layer-sum | 300 | 8 / 32 | 1.450147 | 0.304264 | 1.075623 | +0.042735 | -0.018731 |
| Spectral auxiliary input | 300 | 8 / 32 | 1.594168 | 0.277464 | 1.074935 | +0.015934 | -0.019418 |
| Layer-9 + universal, Postnet reference | 0 | 3 / 3 | 1.244862 | 0.264994 | 1.120857 | +0.000000 | +0.000000 |
| Postnet smoke | 500 | 3 / 3 | 1.243544 | 0.265539 | 1.121324 | +0.000546 | +0.000468 |
| Postnet long run | 3000 | 3 / 3 | 1.241681 | 0.252833 | 1.120732 | -0.012160 | -0.000125 |
| Layer-9 + Approach-A, joint reference | 0 | 3 / not logged | — | 0.281058 | 1.107004 | +0.000000 | +0.000000 |
| Stabilized joint fine-tuning | 500 | 3 / not logged | — | 0.260298 | 1.123442 | -0.020760 | +0.016438 |

The weighted/spectral reference is STOI 0.261529, PESQ 1.094354, L1 1.288527; the tiny difference from this run's step-0 audio scores is retained rather than silently overwritten. Weighted and spectral models were trained from fresh mapper initialization at LR 1e-3, unlike this pretrained-mapper fine-tune at 1e-5. Weighted had no clipping; spectral clipped 295/300 updates. Mel-GAN has better final L1 and PESQ than both, but lower STOI than both. None demonstrates a consistent improvement across quality metrics.

The completed Postnet long run is `resume3000_lr3e4_20260925_201235_42161`: 500 initial updates at 1e-4 followed by 2,500 resumed updates at 3e-4. Its final STOI decline is −0.012160, with all three utterances regressing, and PESQ is essentially unchanged. This supersedes the older Postnet document's “prepared, not executed” status. Merely matching 3,000 optimizer steps would not produce a controlled comparison: Postnet used one utterance/update versus four here, different validation coverage, a different objective and a changed LR.

Other historical joint failures are incompletely measured locally:

| Historical joint attempt | Evaluation | STOI baseline → result | PESQ | Comparable validation L1 |
| --- | --- | --- | --- | --- |
| Earlier long joint, step 2,000 | User-reported TEST | 0.180 → 0.146 | Not available locally | Not available locally |
| Earlier long joint, step 10,000 | User-reported TEST | 0.180 → 0.139 | Not available locally | Not available locally |
| Original local `smoke_100` / `smoke_100_hybrid` | Only step-0 audio validation saved | ≈0.281055 → no post-training score | Baseline ≈1.107004; no final score | Not logged |

The local hybrid attempt had only 35 updates per the stability investigation, not a completed 100-step result. The historical TEST scores are quoted from that investigation, not independently recovered checkpoints, and must not be ranked against this eight-item VAL subset. A three-step stabilized CPU connectivity probe also exists; it is not a quality trial (STOI 0.281058 → 0.280450, PESQ 1.107004 → 1.109381).

## Why every mapper update was clipped

`mapper_clip_rate = 1.0` means **300/300 raw total mapper gradient norms exceeded 1.0**, so standard PyTorch clipping rescaled the gradients before Adam. It does not mean clipping failed or that gradients were nonfinite. The metric measures how often the bound was active, not the severity or trend of instability.

| Phase | Steps | Raw mapper norm min | Mean | Max |
| --- | --- | ---: | ---: | ---: |
| Pure mel-L1 warmup | 1–50 | 8.295467 | 13.552143 | 26.547384 |
| Adversarial ramp | 51–149 | 7.037105 | 13.489802 | 33.382175 |
| Full weight | 150–300 | 7.325335 | 12.743785 | 22.292347 |

Raw norms span **7.037105–33.382175** over the run. The full-weight phase's mean is lower than warmup's; the cumulative maximum remains unchanged from the step-148 validation onward. Post-clipping mapper norms span **0.999970973–1.000033855**, consistent with float32 reduction differences around 1. All logged losses/gradients are finite; 300 updates completed; validation L1 remained bounded near 1.275–1.278 after early adaptation. There is no observed numerical explosion or runaway loss trend. Discriminator clipping activates 49/300 times, with maximum raw norm 4.075301.

The `10 × mel-L1` term scales its gradient by ten. During warmup the measured total gradient is entirely from that term: unweighted mel-L1 norms would be approximately 0.830–2.655. Thus frequent clipping at a threshold of 1.0 is unsurprising even before adversarial pressure exists. During ramp/full weight the logs contain the combined gradient, so they do not establish each loss's separate gradient contribution. Loss magnitudes alone cannot establish gradient dominance. With Adam, clipping also does not directly bound parameter-update norms.

For completeness, the original failed smoke already performed actual clipping, then imposed an excessively strict assertion. It failed on the mapper at **step 4, λ = 0**, with raw norm **26.155202865600586** and post-clipping norm **1.000014305114746**, recovered by the identical-seed rerun (steps 1–3 matched exactly). Its original implementation was:

```python
def clip(parameters):
    parameters = list(parameters)
    raw = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
    post = float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters if p.grad is not None])))
    if post > 1.00001:
        raise RuntimeError('Gradient clipping bound violated')
    return raw, post
```

The fix retained rescaling and the nonfinite hard failure, removed the finite rounding assertion, and added per-component, per-step diagnostics before optimizer updates. The old mapper assertion would have fired on 19 of the successful run's 300 updates. Six targeted tests passed. These observations support **finite gradients consistently above a conservative clip threshold**, not numerical instability. Stable arithmetic nevertheless does not imply improving speech quality.

## Are the GAN losses plateaued?

**No: the adversarial game is still actively changing at step 300.** The mapper's validation mel-L1 is approximately plateaued, but the GAN losses are not.

On the fixed validation set, discriminator loss falls from 0.247931 at step 100 to 0.212667 at 200 and 0.156920 at 300 (a 26.2% drop from 200 to 300). Generator adversarial loss rises from 0.277796 to 0.309885 to 0.546468 (a 76.3% rise from 200 to 300). The denser epoch records reinforce this: D falls 0.197722 → 0.176680 → 0.152108 at steps 222/259/296, while G rises 0.292600 → 0.456639 → 0.485398, then 0.546468 at 300. The small final D rebound does not erase the broader movement.

Training-window averages show the same direction, although their batches differ and generator training loss is evaluated after the D update, unlike fixed-model validation:

| Training steps | Mean D loss | Mean G adversarial loss | Mean mel-L1 |
| --- | ---: | ---: | ---: |
| 151–200 | 0.231287 | 0.282575 | 0.920402 |
| 201–250 | 0.204479 | 0.314700 | 0.907104 |
| 251–300 | 0.174195 | 0.364063 | 0.922148 |

For the implemented least-squares objectives, if D output 0.5 everywhere, both D loss and G adversarial loss would be 0.25. The early neighborhood of 0.25 is therefore compatible with an initially weak/uninformative discriminator, not proof of convergence. Later lower D loss alongside higher G loss is consistent with D becoming better at separating generated from real mels and the mapper failing to keep pace. Aggregate losses alone cannot prove equilibrium, discriminator overfitting, or mode collapse; separate real/fake score distributions and additional diagnostics were not logged.

At step 300, the weighted validation adversarial scalar is about 0.05465 versus 12.75986 for the weighted mel term, roughly 0.43%. This explains the scalar objective's scale, but not the ratio of gradient contributions. Only 151 updates (150–300 inclusive) ran at full adversarial weight. A longer run could therefore produce different dynamics; the current data do not support assuming those changes would improve intelligibility.

## Recommendation and limits of extrapolation

**Do not run the same setup for 2,000–3,000 steps merely because the GAN has not converged or to match Postnet's step count.** The decision is based on absent quality benefit, not the clip rate: final STOI/PESQ both regress, PESQ is below baseline at every trained evaluation, and the late discriminator gains coincide with rising generator adversarial loss and falling STOI. There is no positive late audio trend to extrapolate. The approximately 1% mel-L1 improvement largely appeared before strong adversarial weighting and cannot be attributed to the GAN without a matched L1-only fine-tune control.

This is not proof that more steps cannot help. One seed, eight audio samples, sparse audio evaluations and only 300 updates cannot settle eventual GAN behavior. An unchanged longer run is defensible only as a deliberately bounded diagnostic of whether the mapper eventually responds; it is currently a low-confidence quality bet. A matched L1-only continuation/control, common validation coverage and separate loss-gradient/real-fake diagnostics would be more informative than treating training duration as the missing ingredient. Any future experiment should judge STOI/PESQ and listening against its own baseline, not GAN losses alone. None was launched here.

## Artifact references

All paths below are relative to this document. Numerical tables were generated from the saved JSON records, not transcribed from rounded progress messages.

- mel: [validation records](../results/checkpoints/mel_adversarial_mapper/smoke300_20260925_224308_80333/validation.jsonl), [summary](../results/checkpoints/mel_adversarial_mapper/smoke300_20260925_224308_80333/summary.json).
- weighted: [validation records](../results/checkpoints/weighted_layer_mapper/smoke300_20260925_204353_47079/validation.jsonl), [summary](../results/checkpoints/weighted_layer_mapper/smoke300_20260925_204353_47079/summary.json).
- spectral: [validation records](../results/checkpoints/spectral_aux_mapper/smoke300_20260925_220350_64148/validation.jsonl), [summary](../results/checkpoints/spectral_aux_mapper/smoke300_20260925_220350_64148/summary.json).
- post: [validation records](../results/checkpoints/postnet/smoke500_cpu_20260925_165833_35665/validation.jsonl), [summary](../results/checkpoints/postnet/smoke500_cpu_20260925_165833_35665/summary.json).
- postlong: [validation records](../results/checkpoints/postnet/resume3000_lr3e4_20260925_201235_42161/validation.jsonl), [summary](../results/checkpoints/postnet/resume3000_lr3e4_20260925_201235_42161/summary.json).
- joint: [validation records](../results/checkpoints/joint_finetune/stable_smoke500_cpu_05777c63/validation.jsonl), [summary](../results/checkpoints/joint_finetune/stable_smoke500_cpu_05777c63/summary.json).
- Mel-GAN detailed diagnostics: [epoch validation](../results/checkpoints/mel_adversarial_mapper/smoke300_20260925_224308_80333/epochs.jsonl), [training losses](../results/checkpoints/mel_adversarial_mapper/smoke300_20260925_224308_80333/losses.jsonl), [gradient records](../results/checkpoints/mel_adversarial_mapper/smoke300_20260925_224308_80333/gradients.jsonl), [configuration/provenance](../results/checkpoints/mel_adversarial_mapper/smoke300_20260925_224308_80333/run.json).
- Context: [joint stability investigation](joint_stability_diagnosis.md), [Postnet setup](postnet_smoke.md), [weighted-layer setup](weighted_layer_mapper_smoke.md), [spectral-aux report](spectral_aux_mapper_smoke.md), [mel-GAN implementation notes](mel_adversarial_mapper_smoke.md).

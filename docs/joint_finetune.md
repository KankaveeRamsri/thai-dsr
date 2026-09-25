# Joint mapper and HiFi-GAN fine-tuning

New entry point: `python -m src.training.joint_finetune`. No prior source, checkpoint, or log is modified. Default execution is a 100-step smoke test, not an overnight run.

## Initialization and data

The mapper is resolved from `results/audio_compare/w6_vocoder_ft/provenance.json` and its SHA-256 is checked. This selects `results/checkpoints/mapper_layer9.pt` (validation L1 1.2885), rather than the older inference CLI default `best_mapper.pt` (1.3504). Override with `--mapper-checkpoint`.

`--vocoder-init thai` loads `hifigan_thai/g_00010000` and MPD/MSD from `hifigan_thai/do_latest`, requiring its step to equal 10000. Optimizers start fresh at the joint learning rates. `--vocoder-init universal` loads `UNIVERSAL_V1/g_02500000`; absent an accompanying discriminator checkpoint, MPD/MSD start randomly. Use `--discriminator-checkpoint` to supply a compatible state explicitly. The universal alternative therefore needs its own short stability check.

The authoritative paired manifest is `data/manifest_w5.csv`, severity `severe`, and `data/splits_w5.json`: 147 train, 32 val, 31 test. Split IDs are verified disjoint and complete. Test audio is not evaluated or encoded by training. Clean-only HiFi-GAN caches are not sufficient for the distorted inputs, so the paired manifest is used directly.

## Forward and losses

One utterance per step. Frozen layer-9 embeddings are computed on demand from distorted audio and cached in RAM. Whole-utterance frozen embeddings are interpolated to clean time, then a 64-frame target window plus 32 context frames on each side goes through the trainable BiLSTM. Boundary context uses replicated embeddings. Only the central 16384-sample (0.743 s) predicted-mel / clean-waveform crop is used for waveform losses. This bounds MPS backward graph size; validation runs the mapper over the full utterance. Shorter utterances use their full available length, without padding.

Training linearly interpolates input embeddings to the clean target's frame count, as existing mapper training does. This is only approximate alignment: local tempo changes are not corrected by DTW. Validation uses distorted input duration, without clean-length oracle information, so this remaining train/inference timing mismatch is visible in validation scores. Joint training alone is not proven to solve that mismatch.

The mapper's existing `[-11.512925, 2]` clamp and `(B,T,80)->(B,80,T)` conversion are retained. Gradients pass through unclipped bins. The generator keeps weight normalization while training.

Loss = `1 * adversarial + 2 * feature_matching_raw + 45 * mel + 2 * content`.

- Adversarial: vendor LSGAN MPD + MSD.
- Feature matching: vendor function already applies x2; the script divides by 2 to log raw loss, then applies configurable x2 exactly once.
- Mel: existing differentiable `compute_mel_tensor`, HiFi-GAN loss configuration (`fmax=None`), real versus generated waveform. CPU STFT avoids MPS complex-operator issues while retaining gradients through device transfers.
- Content: L1 of generated versus clean layer-9 XLSR features, using the same frozen model. Both waveform branches use differentiable 22.05->16 kHz resampling and feature-extractor-equivalent mean/variance normalization. Only clean/input branches run under `no_grad`; the generated branch MUST retain autograd. Content weight 2 is a tunable starting point (try 1-5). Short crops limit phonetic context; longer crops require more memory.

The encoder is always eval/frozen. Default `--content-device cpu` runs both input and differentiable perceptual passes on CPU; mapper, generator and discriminators run on MPS. Device transfers preserve waveform gradients. An initial all-MPS attempt exhausted practical unified-memory headroom on this 8 GiB machine after six steps (6.8 GiB peak physical footprint, heavy swapping) and was interrupted; its logs are preserved. This is why the hybrid CPU/MPS configuration is the default. Unused upper transformer blocks are omitted in memory; block 10 is retained so `hidden_states[9]` remains the exact original pre-block-10 representation, including stable-layer-norm semantics. Startup verifies bit-identical layer-9 output against the full model. No encoder weights are updated. Reference: https://huggingface.co/docs/transformers/v5.10.2/en/model_doc/wav2vec2

Learning rates: mapper 1e-5, generator 1e-5, discriminators 1e-5; AdamW betas from the vocoder config (0.8, 0.99), weight decay 0. Previous standalone discriminator LR was 2e-4. Gradient norm clipping at 100 protects each optimizer group; logged norms are pre-clipping. All settings are CLI parameters.

The first training step independently verifies nonzero finite adversarial/content gradients to mapper output. Every step checks mapper, generator, and discriminator gradient norms, finite losses, and absence of gradients in frozen wav2vec2 parameters.

## Artifacts and validation

Each new run creates an exclusive subdirectory of `results/checkpoints/joint_finetune/`. It refuses an existing run directory and refuses output outside that tree. Joint snapshots atomically include mapper, generator, both discriminators, both optimizers, step, settings, initialization provenance, and sampling/CPU RNG state. No automatic checkpoint deletion occurs. `--resume PATH --output-dir NEW_DIRECTORY --max-steps TOTAL` restores joint states to a new run; this is not a bitwise replay guarantee for accelerator randomness.

`losses.jsonl` records every loss and gradient norm per step. `validation.jsonl` records per-utterance and mean STOI/PESQ for the same three validation IDs at step 0 and intervals; generated validation WAVs are saved alongside. Validation uses existing shared-length truncation, without gain/time alignment. `gradient_checks.json`, `run.json`, and `summary.json` provide audit information.

## Overnight launcher (prepared, not launched)

```sh
bash scripts/run_overnight_joint_finetune.sh
```

Defaults: 2000 steps, checkpoint every 500, validation every 250, three validation utterances. Atomic `mkdir` lock; duplicate process guard; timestamped `mktemp` log; detached `nohup` + `disown`; `caffeinate -i -m`; wait for training and assertion process exit; sync; `pmset sleepnow` after completion, including nonzero training exit. Like the original launcher, automatic sleep is cancelled if startup cannot be verified or another training process is present. The launcher itself has only been syntax-checked; it has not been executed.

```sh
JOINT_MAX_STEPS=2000 JOINT_CONTENT_WEIGHT=2 bash scripts/run_overnight_joint_finetune.sh
# Alternative initialization; first do a separate smoke test:
JOINT_VOCODER_INIT=universal bash scripts/run_overnight_joint_finetune.sh
```

Full joint snapshots consume substantial disk space. The launcher budgets 1400 MiB per snapshot plus 2 GiB reserve and refuses insufficient disk space. Default 2000 steps is deliberate given this machine's current free space. The trainer also requires 2 GiB free before each snapshot.

Contract checks: `MPLCONFIGDIR=/tmp/thai-dsr-mpl .venv/bin/python -m unittest tests.test_joint_finetune -v`.

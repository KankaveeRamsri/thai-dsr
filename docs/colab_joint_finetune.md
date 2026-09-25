# Colab joint fine-tuning handoff

Open `scripts/colab_joint_finetune.ipynb` in Colab, select a GPU runtime, and run cells in order. Upload the local `scripts/colab_transfer_bundle.zip` to **My Drive/thai_dsr_colab/**. The notebook expects `/content/drive/MyDrive/thai_dsr_colab/colab_transfer_bundle.zip`.

The archive is **1,290,213,395 bytes (1.202 GiB / 1,230.44 MiB)**. Paths inside the archive are relative to the repository root. `scripts/colab_bundle_manifest.json` lists every file, byte size, and SHA-256; the notebook checks them after extraction.

| Bundled path | Bytes | Notes |
| --- | ---: | --- |
| `data/manifest_w5.csv` | 43,557 | Already tracked in git; also bundled as requested |
| `data/splits_w5.json` | 4,234 | Already tracked in git; also bundled as requested |
| `data/hifigan_finetune/` | 106,228,857 | 421 cached files; joint training itself reads paired audio |
| `results/checkpoints/mapper_layer9.pt` | 154,640,271 | Current pipeline mapper; validation L1 1.2885; selected by existing provenance |
| `results/checkpoints/hifigan_thai/g_00010000` | 55,822,075 | Thai generator initialization |
| `results/checkpoints/hifigan_thai/do_latest` | 960,636,825 | Required MPD/MSD initialization at step 10000; standalone optimizer states also present |
| `results/checkpoints/hifigan_thai/config.json` | 799 | Required generator configuration |
| `vendor/hifi-gan/checkpoints/UNIVERSAL_V1/config.json` | 799 | Universal configuration |
| `vendor/hifi-gan/checkpoints/UNIVERSAL_V1/g_02500000` | 55,788,858 | Universal generator alternative |
| `data/raw/` | 1,600,440 | 10 clean audio files referenced by severe manifest rows |
| `data/distorted/` | 1,700,988 | 10 corresponding severe distorted files |
| `data/common_voice/` | 27,750,496 | 200 clean audio files |
| `data/distorted_cv/` | 29,330,706 | 200 severe distorted audio files |
| `vendor/hifi-gan/` (source only) | 2,822,073 | 16 upstream tracked files, including license; excludes nested git metadata |

The vendor source snapshot is revision `4769534d45265d52a904b850da5a622601885777` from https://github.com/jik876/hifi-gan. All 210 pairs are present because the loader checks file existence across all splits; training uses only train and validation audio.

An additional public model, `airesearch/wav2vec2-large-xlsr-53-th`, is downloaded automatically from Hugging Face by the notebook (roughly 1.2 GiB weights, plus small configuration files). It is not in git or the zip; the local Mac Hugging Face cache is not needed. Colab needs network access for this download and pip. No precomputed embeddings are required.

The notebook installs requirements plus explicit PyYAML and pins Transformers to the local smoke environment's 5.13.1. It preserves Colab's matching torch/torchaudio versions with pip constraints. Training runs in fresh subprocesses. Existing local training/inference sources were not changed for Colab; CUDA is already supported. CPU STFT/resampling remain as implemented.

A 50-step smoke test reports training steps/sec, total throughput including startup/validation/checkpoint writes, and estimated 2000-step compute time. The full run starts from the original mapper/Thai vocoder initialization, saves every 250 steps, validates every 250 steps, and targets 2000 total steps. Both smoke and full snapshots go directly to Drive through a symlink at `results/checkpoints/joint_finetune`. No local-copy checkpoint interval is involved. The checkpoint includes mapper, generator, MPD/MSD, both optimizers, CPU and sampling RNG states.

On restart, rerun setup and extraction; smoke skips if a full checkpoint exists. The full cell selects the highest completed step in `checkpoints/full_*/joint_*.pt`, validates its state, and resumes into a new run directory. Temporary incomplete writes are ignored. An explicit `RESUME_PATH` can select a prior intact snapshot. CUDA RNG is not captured by the existing trainer, so resume is not a bitwise replay guarantee. About 12 GiB free Drive space is recommended for the default run and bundle; snapshots are never deleted automatically.

Validation performed locally: eight existing joint/dataset/checkpoint/mel tests passed; all notebook code cells compile. CUDA execution and Google Drive mounting must be verified in Colab because this Mac has no CUDA GPU. The existing hybrid smoke configuration is preserved in `configs/joint_smoke_hybrid.json`; this is provenance, not a trainer config-file argument.

# Pipeline

This document describes the end-to-end Thai DSR (Dysarthric Speech
Reconstruction) pipeline, from raw data to a trained reconstruction model
and evaluated output.

## Overview

1. **Data collection** — raw recordings land in `data/raw/`.
2. **Preprocessing** (`src/preprocessing/`)
   - `distortion.py` generates synthetic dysarthric-like distortions from
     clean audio into `data/distorted/` (used when paired real dysarthric
     data is limited).
   - `extract_embedding.py` extracts speaker/content embeddings into
     `data/embeddings/`.
   - `build_manifest.py` produces the dataset manifest used for training.
3. **Modeling** (`src/models/`)
   - `encoder.py` — speaker/content encoders.
   - `mapper.py` — distorted-to-clean feature mapping network.
   - `vocoder.py` — feature-to-waveform synthesis.
4. **Training** (`src/training/`) — `train.py` fits the mapper (and
   optionally fine-tunes encoder/vocoder) using `dataset.py` and the
   configs in `configs/`.
5. **Inference** (`src/inference/run.py`) — runs the trained pipeline on
   new dysarthric audio to produce reconstructed speech.
6. **Evaluation** (`src/evaluation/`)
   - `metrics.py` — STOI/PESQ objective quality scores.
   - `wer.py` — ASR-based word error rate scoring.

## TODO

- Fill in dataset sources and licensing.
- Document distortion types used and their rationale.
- Document final model architecture choices once finalized.

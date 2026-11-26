# CV 27.0 test evaluation: W2 systems, with ASR intelligibility

This folder holds the full test set: 308 clips × mild/moderate/severe, 48 speakers.

| file | what it is |
|---|---|
| `summary.md`, `summary_by_severity.csv`, `speaker_similarity.csv` | All metrics by severity, with speaker-level bootstrap 95% CIs (`scripts/summarize_cv_eval.py`, 10,000 resamples, seed 42). |
| `paired_*.csv` | Paired comparisons per severity: b − a mean difference with CI, share of clips where b is better, and Wilcoxon p-values. The per-speaker p-value is the one to report. |
| `asr/<system>.csv` | Whisper transcript and per-clip CER/WER (with edit counts) for each system. `clean` has one row per clip. |
| `asr/<system>_run_info.json` | ASR provenance. |

Per-clip audio and per-utterance tables for the other metrics are kept outside the repo, in
`<data root>/eval/test_full/`.

## ASR setup (pinned)

- **Model:** `openai/whisper-large-v3` at revision `06f233fe06e710322aca913c1bc4249a0d71fce1`. Runs on CPU in float32; on MPS, float16 and bfloat16 decode garbage.
- **Decoding:** greedy, `language=th`, `task=transcribe`, at most 128 new tokens.
- **Text normalisation:** `thai_text-v1` (`src/evaluation/thai_text.py`), with PyThaiNLP 5.4.0 and the newmm tokenizer for WER.
- **Run:** all five systems used the same settings fingerprint `bb1621a0bf26c0a1` and code at commit 328f122 (clean tree).
  - 50 of the clean transcripts and 6 of the distorted ones come from the sanity and resume tests, which used the same settings.
- **Scoring:** each system's own output (native timeline) is scored against the Common Voice sentence.
- **CER/WER definition:** corpus-level, i.e. Σ edits / Σ reference length.

## Results: corpus CER (primary) and WER, with speaker-level 95% CI

| system | mild CER | moderate CER | severe CER | all CER | all WER |
|---|---|---|---|---|---|
| clean | 0.057 [0.048, 0.068] | – | – | 0.057 | 0.158 [0.137, 0.181] |
| distorted (floor) | 0.236 [0.199, 0.277] | 0.483 [0.427, 0.546] | 0.656 [0.603, 0.714] | 0.459 | 0.782 |
| copysynth_tempo (ceiling) | 0.084 [0.069, 0.101] | 0.087 [0.074, 0.102] | 0.102 [0.087, 0.118] | 0.091 | 0.226 |
| mapper_tempo (W2 b) | 0.537 [0.495, 0.578] | 0.645 [0.606, 0.686] | 0.729 [0.683, 0.778] | 0.637 | 1.015 |
| mapper_linear_deploy (W2 a) | 1.010 [0.955, 1.076] | 0.967 [0.933, 1.006] | 0.971 [0.928, 1.025] | 0.983 | 1.071 |

**mapper_tempo is worse than the distorted input at every severity.** CER difference
(tempo − distorted), with speaker-level CI and Wilcoxon p:

| severity | CER difference [95% CI] | p (speakers) |
|---|---|---|
| mild | +0.300 [0.272, 0.330] | < 1e-4 |
| moderate | +0.162 [0.110, 0.212] | < 1e-4 |
| severe | +0.073 [0.006, 0.135] | 0.0066 |

The gap narrows as the distortion gets stronger, but it never crosses zero.

**Tempo vs linear.** The tempo model beats linear on CER at every severity (all
severities −0.346 [−0.399, −0.294]). On WER it wins only at mild; at severe it is
worse, +0.068 [0.004, 0.135], p = 0.048.

## Why WER and CER disagree at severe

At severe, tempo is worse than linear on WER but better on CER. The decomposition on
the 308 severe clips explains why:

| | mapper_tempo | mapper_linear_deploy |
|---|---|---|
| hypothesis / reference length, characters | 1.01 | 0.49 |
| hypothesis / reference length, words (newmm) | 1.26 | 0.47 |
| characters per word, hypothesis (reference is 4.37) | 3.51 | 4.54 |
| CER: substitution / deletion / insertion | 0.51 / 0.11 / 0.11 | 0.33 / 0.57 / 0.07 |
| WER: substitution / deletion / insertion | 0.77 / 0.04 / 0.30 | 0.42 / 0.58 / 0.05 |
| hypotheses that are a Whisper stock phrase | ~0% | 83%: "โปรดติดตามตอนต่อไป" 461×, "โอ้" 88×, "โอ้โห" 60× (all severities) |

- **linear.** Whisper hears no speech and outputs short stock hallucinations. Most
  reference words are deleted. A deletion costs at most 1 per reference word, so WER
  sits near 1.0 even though almost nothing is recognised (CER ≈ 0.97).
- **tempo.** Whisper hears speech-like audio and outputs Thai of the right length, but
  with the wrong words: phonetically close substitutions and non-words. newmm splits
  these non-words into short fragments, for example "อุณห|ภิกษ์", "ปุ|กรัน", "ท|ริ|ศ".
  The hypothesis therefore has about 25% more "words" than the reference, and the
  extras count as insertions, which pushes WER above 1.
  - Whisper's own runaway insertions are rare: about 2% of clips have a hypothesis
    longer than twice the reference.
- **Conclusion.** The WER reversal is an artefact of word tokenization on garbled
  text, combined with WER's cap for deletions. It is not a sign that tempo
  recognises less. The errors themselves are real (tempo output is not
  intelligible to Whisper), and CER, the primary metric, shows them.
- **Same pattern elsewhere.** The distorted input shows the same signature
  (words/characters ratio 1.24, WER 1.02 at severe).

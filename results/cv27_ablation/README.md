# Fixed-layer ablation: results

The plan is pre-registered in `docs/ablation_plan.md`, together with Amendment 1 and
Deviations 2 and 3.

**Training.** Kaggle, Tesla T4, commit b6c04e3, 8 h 42 min (`kaggle_summary.json`,
`run_info_layer<NN>.json`). All six runs are ok, with 50 epochs and input normalisation
from the train split (2,383 clips).

| layer | best epoch | best val loss |
|---|---|---|
| 3 | 27 | 1.0501 |
| 6 | 20 | 1.0908 |
| 9 | 24 | 1.0978 |
| 12 | 17 | 1.1347 |
| 18 | 18 | 1.1613 |
| 24 | 12 | 1.2101 † |

**Evaluation.** The full CV 27.0 test set: 924 rows (308 clips × 3 severities), 48
speakers. The protocol is the one used before (`scripts/evaluate_cv_aligned.py`, melwarp),
with test embeddings extracted on the Mac (`scripts/run_ablation_eval.sh`).
`scripts/ablation_tests.py` produced `ablation_results.md` and the CSVs here.

## ⚠ Layer 24 is invalid in these files (Deviation 3)

On Kaggle, transformers 5.20.2 returned block 24 **without** the final LayerNorm as
`hidden_states[24]`. The checkpoint (†) was therefore trained on pre-LN outputs.

It was evaluated with post-LN Mac test embeddings. Its rows in `layer_table.csv` and
`exploratory_vs_layer9.csv` come from that mismatched input and must not be used. This
covers its near-zero speaker similarity and F0 correlation, and its HNR of 29 dB. **H2
for layer 24 is not decided**, even though the generated table says "supported".

Layers 3–18 are unaffected: their training statistics match the Mac test embeddings,
and layer 9 matches the W2 archive to 4.6e-6.

## Pre-registered tests (valid part)

| hypothesis | layer | result |
|---|---|---|
| H1 (lower layers keep F0 and speaker identity better than layer 9) | 3 | **supported**: f0_corr +0.065 [0.036, 0.095], f0_rmse_st −0.38 st [−0.58, −0.18], spk_sim_other +0.021 [0.014, 0.029]; Holm p ≤ 0.003 |
| H1 | 6 | **not supported**: all three differences are near 0, Holm p ≥ 0.59 |
| H2 (deeper layers lose speaker identity) | 18 | **supported**: spk_sim_other −0.039 [−0.045, −0.032], Holm p < 1e-12 |
| H2 | 24 | **not decided** (invalid evaluation, see above) |

Seed rule:
- Layer 3's spk_sim_other difference (+0.0208) is just above the 0.02 threshold
  that triggers a seed-43 re-run, so no re-run is required. Its CI does reach below
  0.02 (lower bound 0.0135).
- Layer 18 (−0.039) is well clear of the threshold.

Exploratory (raw p, not corrected):
- **Intelligibility.** Layer 3 is not less intelligible than layer 9 (stoi_melwarp
  +0.005 [0.002, 0.008]), so the trade-off anticipated in H1 does not appear.
- **Depth.** Layers 12 and 18 are worse than layer 9 on every metric.
- **Input normalisation.** The retrained layer 9 (input normalisation) beats the W2
  layer 9 on speaker similarity (0.235 vs 0.203), HNR (7.3 vs 4.8 dB) and F0 RMSE
  (4.6 vs 5.1 st). That comparison is exploratory under Amendment 1.

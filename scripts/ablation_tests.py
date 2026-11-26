#!/usr/bin/env python3
"""Pre-registered H1/H2 tests and the layer x metric table for the fixed-layer ablation.

Implements "Pre-registered hypotheses" and "Test procedure" of
docs/ablation_plan.md (with Amendment 1) exactly:

  H1  layers 3 and 6 vs layer 9: higher f0_corr, lower f0_rmse_st, higher
      spk_sim_other (6 tests, one Holm family)
  H2  layers 18 and 24 vs layer 9: lower spk_sim_other (2 tests, one Holm family)

Layer 9 is the retrained layer 9 (mapper_tempo_norm_layer09), not the W2 run.
For each (layer, metric): paired per-clip differences layer L minus layer 9,
pooled over the three severities (924 pairs, minus pairs where either side is
NaN); a two-sided Wilcoxon signed-rank test on the 48 per-speaker mean
differences; a 95% speaker-clustered bootstrap CI (10,000 resamples, seed 42).
Holm-Bonferroni within each family at alpha 0.05.

Decision per (layer, metric):
  supported      Holm p < 0.05 and the mean difference is in the predicted direction
  contradicted   Holm p < 0.05 and the opposite direction
  not supported  otherwise
H1 is supported for a layer only if spk_sim_other and at least one F0 metric
are supported. Seed rule: a supported or contradicted result whose difference is
smaller than 0.02 (cosine, spk_sim_other) is flagged for a seed-43 re-run before
any claim. The plan gives no threshold for the F0 metrics.

Also written: the layer x metric table (all severities pooled, mean [95% CI]),
with floor, ceilings and the W2 run as reference rows; and exploratory
differences vs layer 9 for every layer and metric (raw p, no correction; the
H1 intelligibility trade-off on stoi_melwarp is reported here).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import cv_data_root  # noqa: E402

_spec = importlib.util.spec_from_file_location("summarize_cv_eval", ROOT / "scripts/summarize_cv_eval.py")
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)

LAYERS = [3, 6, 9, 12, 18, 24]
REFERENCE_LAYER = 9
TABLE_METRICS = ["stoi_melwarp", "pesq_melwarp", "mel_l1_clean", "spk_sim_same", "spk_sim_other",
                 "f0_corr", "f0_rmse_st", "vuv_agree", "hnr_db"]
LOWER = {"mel_l1_clean", "f0_rmse_st"}
FAMILIES = {
    "H1": [(layer, metric, direction) for layer in (3, 6)
           for metric, direction in (("f0_corr", "higher"), ("f0_rmse_st", "lower"), ("spk_sim_other", "higher"))],
    "H2": [(layer, "spk_sim_other", "lower") for layer in (18, 24)],
}
SEED_RULE = {"spk_sim_other": 0.02, "spk_sim_same": 0.02, "stoi_melwarp": 0.01}
REFERENCE_ROWS = [("distorted", "floor (distorted input)"), ("copysynth_tempo", "ceiling (copy synthesis, tempo)"),
                  ("copysynth_clean", "ceiling (copy synthesis, clean)"), ("mapper_tempo", "W2 layer 9 (no input norm)")]


def system_name(layer):
    return f"mapper_tempo_norm_layer{layer:02d}"


def load(eval_dir, name):
    path = eval_dir / name / "per_utterance.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    pitch = eval_dir / name / "pitch_metrics.csv"
    if "f0_corr" not in df.columns and pitch.exists():
        df = df.merge(pd.read_csv(pitch), on=["utterance_id", "severity"], how="left", validate="one_to_one")
    for metric in TABLE_METRICS:
        if metric not in df.columns:
            df[metric] = np.nan
    return df.set_index(["utterance_id", "severity"]).sort_index()


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def paired(a, b, metric, n_boot, seed):
    """b - a per clip: (n, mean diff, CI low, CI high, Wilcoxon p on per-speaker means, n speakers)."""
    frame = pd.DataFrame({"a": a[metric], "b": b.loc[a.index, metric], "client_id": a["client_id"]}).dropna()
    diff = frame["b"] - frame["a"]
    n, mean, low, high = summarize.cluster_bootstrap(diff, frame["client_id"], n_boot, seed)
    speaker = diff.groupby(frame["client_id"]).mean()
    return n, mean, low, high, summarize.safe_wilcoxon(speaker), len(speaker)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", type=Path, default=None, help="default: <data root>/eval/test_full")
    parser.add_argument("--out", type=Path, default=None, help="default: <eval-dir>/summary_ablation")
    parser.add_argument("--expected-rows", type=int, default=924)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    eval_dir = args.eval_dir or cv_data_root() / "eval" / "test_full"
    out = args.out or eval_dir / "summary_ablation"
    out.mkdir(parents=True, exist_ok=True)

    frames = {}
    for layer in LAYERS:
        df = load(eval_dir, system_name(layer))
        if df is None:
            raise SystemExit(f"missing {eval_dir / system_name(layer)}: evaluate every layer first")
        frames[system_name(layer)] = df
    for name, _ in REFERENCE_ROWS:
        df = load(eval_dir, name)
        if df is not None:
            frames[name] = df
    keys = frames[system_name(REFERENCE_LAYER)].index
    for name, df in frames.items():
        if args.expected_rows and len(df) != args.expected_rows:
            raise SystemExit(f"{name}: {len(df)} rows, expected {args.expected_rows}")
        if not df.index.equals(keys):
            raise SystemExit(f"{name} covers different (clip, severity) rows")

    # Layer x metric table, all severities pooled.
    rows = [(name, role) for name, role in REFERENCE_ROWS if name in frames]
    rows += [(system_name(layer), f"layer {layer}") for layer in LAYERS]
    table = []
    for name, role in rows:
        df = frames[name]
        record = {"system": name, "row": role}
        for metric in TABLE_METRICS:
            n, mean, low, high = summarize.cluster_bootstrap(df[metric], df["client_id"], args.n_boot, args.seed)
            record.update({metric: mean, f"{metric}_low": low, f"{metric}_high": high})
        table.append(record)
    table = pd.DataFrame(table)
    table.to_csv(out / "layer_table.csv", index=False)

    # Pre-registered tests.
    reference = frames[system_name(REFERENCE_LAYER)]
    tests = []
    for family, items in FAMILIES.items():
        results = []
        for layer, metric, direction in items:
            n, mean, low, high, p, n_spk = paired(reference, frames[system_name(layer)], metric, args.n_boot, args.seed)
            results.append({"family": family, "layer": layer, "metric": metric, "predicted": direction, "n_pairs": n,
                            "n_speakers": n_spk, "mean_layer9": float(reference[metric].mean()),
                            "mean_layer": float(frames[system_name(layer)][metric].mean()),
                            "diff": mean, "ci_low": low, "ci_high": high, "p": p})
        for record, p_holm in zip(results, holm([r["p"] for r in results])):
            record["p_holm"] = float(p_holm)
            observed = "higher" if record["diff"] > 0 else "lower"
            if p_holm < 0.05:
                record["decision"] = "supported" if observed == record["predicted"] else "contradicted"
            else:
                record["decision"] = "not supported"
            threshold = SEED_RULE.get(record["metric"])
            record["seed43_rerun_needed"] = bool(record["decision"] != "not supported" and threshold is not None
                                                 and abs(record["diff"]) < threshold)
        tests += results
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "preregistered_tests.csv", index=False)
    verdicts = []
    for layer in (3, 6):
        part = tests[(tests.family == "H1") & (tests.layer == layer)].set_index("metric")["decision"]
        ok = part["spk_sim_other"] == "supported" and "supported" in (part["f0_corr"], part["f0_rmse_st"])
        verdicts.append(("H1", layer, "supported" if ok else "not supported"))
    for layer in (18, 24):
        verdicts.append(("H2", layer, tests[(tests.family == "H2") & (tests.layer == layer)]["decision"].iloc[0]))

    # Exploratory: every layer and metric vs layer 9 (raw p).
    exploratory = []
    for layer in LAYERS:
        if layer == REFERENCE_LAYER:
            continue
        for metric in TABLE_METRICS:
            n, mean, low, high, p, n_spk = paired(reference, frames[system_name(layer)], metric, args.n_boot, args.seed)
            exploratory.append({"layer": layer, "metric": metric, "better": "lower" if metric in LOWER else "higher",
                                "diff": mean, "ci_low": low, "ci_high": high, "p_raw": p, "n_pairs": n})
    exploratory = pd.DataFrame(exploratory)
    exploratory.to_csv(out / "exploratory_vs_layer9.csv", index=False)

    def cell(r, m, digits=3):
        return summarize.fmt(r[m], r[f"{m}_low"], r[f"{m}_high"], digits)

    lines = ["# Fixed-layer ablation: results (docs/ablation_plan.md)", "",
             f"Full CV 27.0 test set ({len(keys)} rows = 308 clips x 3 severities, 48 speakers), all severities "
             f"pooled. Mean [95% CI], speaker-clustered bootstrap ({args.n_boot} resamples, seed {args.seed}).", "",
             "## Layer x metric", "",
             "| row | " + " | ".join(TABLE_METRICS) + " |", "|" + "---|" * (len(TABLE_METRICS) + 1)]
    for _, r in table.iterrows():
        lines.append(f"| {r['row']} | " + " | ".join(cell(r, m) for m in TABLE_METRICS) + " |")
    lines += ["", "Floor: no melwarp (no mel to vocode); its waveform inverse-warp STOI is in the main summary.",
              "Lower is better for mel_l1_clean and f0_rmse_st; hnr_db is best when close to the clean clips' value.",
              "", "## Pre-registered tests (layer L minus retrained layer 9, Holm within family)", "",
              "| family | layer | metric | predicted | diff [95% CI] | p (speakers) | p Holm | decision | seed-43 rule |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        lines.append(f"| {r.family} | {r.layer} | {r.metric} | {r.predicted} | "
                     f"{summarize.fmt(r['diff'], r.ci_low, r.ci_high, 4)} | {r.p:.3g} | {r.p_holm:.3g} | "
                     f"**{r.decision}** | {'re-run needed' if r.seed43_rerun_needed else '–'} |")
    lines += ["", "Hypothesis-level verdicts:", ""]
    lines += [f"- {h}, layer {layer}: **{verdict}**" for h, layer, verdict in verdicts]
    lines += ["", "## Exploratory: every layer vs layer 9 (raw p, not corrected)", "",
              "| layer | metric | better | diff [95% CI] | p (speakers) |", "|---|---|---|---|---|"]
    for _, r in exploratory.iterrows():
        lines.append(f"| {r.layer} | {r.metric} | {r.better} | {summarize.fmt(r['diff'], r.ci_low, r.ci_high, 4)} | "
                     f"{r.p_raw:.3g} |")
    (out / "ablation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

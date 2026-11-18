#!/usr/bin/env python3
"""Tables for the CV 27.0 test evaluation written by scripts/evaluate_cv_aligned.py.

Reads <eval-dir>/<system>/per_utterance.csv for each system and writes, in
<eval-dir>/summary/:

  summary_by_severity.csv  long format: system, role, timeline, severity
                           (mild/moderate/severe/all), metric, n, mean,
                           ci_low, ci_high
  summary.md               the primary, sensitivity, speaker and legacy tables
                           as "mean [95% CI]"
  speaker_similarity.csv   spk_sim_same / spk_sim_other per system and
                           severity, plus two clean references: another clean
                           clip of the same client_id, and clean clips of
                           other test speakers
  paired_<b>_vs_<a>_per_clip.csv  one row per (clip, severity): a, b, b - a
  paired_<b>_vs_<a>.csv    per severity and metric: means, mean difference
                           with 95% CI, share of clips where b is better, and
                           Wilcoxon signed-rank p-values. Default pairs: (b)
                           tempo vs (a) linear with the oracle length, and vs
                           (a) linear with the deployable distorted length.

The main comparison is stoi_melwarp / pesq_melwarp (mel-domain inverse warp);
the waveform inverse-warp stoi / pesq are secondary (see evaluate_cv_aligned.py).

Confidence intervals: percentile bootstrap over SPEAKERS (client_id), not
clips, because the 308 test clips come from 48 speakers with 4-8 clips each
and the three severities share clips; resampling clips would treat correlated
rows as independent and give intervals that are too narrow. 10,000 resamples,
seed 42, the same resamples for every system.
Paired tests: p_wilcoxon_speaker uses the per-speaker mean difference (48
independent pairs, the one to report); p_wilcoxon_clip uses every clip and
ignores the speaker clustering (shown for reference only).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import cv_data_root  # noqa: E402

SEVERITIES = ["mild", "moderate", "severe"]
SYSTEMS = [  # (folder, role)
    ("distorted", "floor"),
    ("pv_roundtrip", "control: inverse warp only"),
    ("copysynth_clean", "ceiling (clean timeline)"),
    ("copysynth_tempo", "ceiling (distorted timeline)"),
    ("mapper_linear_oracle", "(a) linear, clean length (oracle)"),
    ("mapper_linear_deploy", "(a) linear, distorted length (deployable)"),
    ("mapper_tempo", "(b) tempo"),
]
PAIRS = [("mapper_tempo", "mapper_linear_oracle"), ("mapper_tempo", "mapper_linear_deploy")]
GROUPS = {
    "main: STOI/PESQ vs untouched clean, mel-domain inverse warp": ["stoi_melwarp", "pesq_melwarp"],
    "secondary: STOI/PESQ vs untouched clean, waveform inverse warp (phase vocoder)": ["stoi", "pesq"],
    "mel L1": ["mel_l1", "mel_l1_clean"],
    "speaker similarity (ECAPA cosine)": ["spk_sim_same", "spk_sim_other"],
    "legacy": ["stoi_trunc", "pesq_trunc", "stoi_fwd", "pesq_fwd"],
}
METRICS = [m for group in GROUPS.values() for m in group]
LOWER_IS_BETTER = {"mel_l1", "mel_l1_clean"}
# mel_l1 is measured against each model's own training target, which differs
# between the clean and distorted timelines, so it is not compared pairwise.
PAIRED_METRICS = ["stoi_melwarp", "pesq_melwarp", "stoi", "pesq", "mel_l1_clean",
                  "spk_sim_same", "spk_sim_other", "stoi_trunc", "pesq_trunc"]


def cluster_bootstrap(values, clusters, n_boot, seed):
    """Mean and percentile 95% CI, resampling clusters with replacement."""
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    values, clusters = values[keep], np.asarray(clusters)[keep]
    if len(values) == 0:
        return 0, np.nan, np.nan, np.nan
    codes, _ = pd.factorize(clusters, sort=True)
    sums = np.bincount(codes, weights=values)
    counts = np.bincount(codes).astype(float)
    k = len(sums)
    weights = np.random.default_rng(seed).multinomial(k, np.full(k, 1.0 / k), size=n_boot)
    means = (weights @ sums) / (weights @ counts)
    return len(values), float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def safe_wilcoxon(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2 or np.all(diff == 0):
        return float("nan")
    return float(wilcoxon(diff).pvalue)


def by_severity(df):
    for severity in SEVERITIES:
        yield severity, df[df["severity"] == severity]
    yield "all", df


def fmt(mean, low, high, digits=3):
    if not np.isfinite(mean):
        return "–"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def markdown_table(summary, metrics, systems):
    head = "| system | severity | " + " | ".join(metrics) + " |"
    lines = [head, "|" + "---|" * (len(metrics) + 2)]
    for name, role in systems:
        for severity in [*SEVERITIES, "all"]:
            cells = []
            for metric in metrics:
                row = summary[(summary["system"] == name) & (summary["severity"] == severity)
                              & (summary["metric"] == metric)]
                cells.append(fmt(*row[["mean", "ci_low", "ci_high"]].iloc[0]) if len(row) else "–")
            lines.append(f"| {name} ({role}) | {severity} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", type=Path, default=None, help="default: <data root>/eval/test_full")
    parser.add_argument("--pair", nargs=2, action="append", metavar=("B", "A"),
                        help="paired comparison B - A; repeatable (default: mapper_tempo vs "
                             "mapper_linear_oracle and vs mapper_linear_deploy)")
    parser.add_argument("--expected-rows", type=int, default=924,
                        help="rows every system must have (0: skip the check, e.g. smoke tests)")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    eval_dir = args.eval_dir or cv_data_root() / "eval" / "test_full"
    out = eval_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)

    frames, systems, keys = {}, [], None
    for name, role in SYSTEMS:
        path = eval_dir / name / "per_utterance.csv"
        if not path.exists():
            print(f"missing {path}: skipped")
            continue
        df = pd.read_csv(path)
        if args.expected_rows and len(df) != args.expected_rows:
            raise SystemExit(f"{path}: {len(df)} rows, expected {args.expected_rows} (use --expected-rows 0)")
        these = set(zip(df["utterance_id"], df["severity"]))
        if keys is not None and these != keys:
            raise SystemExit(f"{name} covers different (clip, severity) rows than {systems[0][0]}")
        keys = these
        frames[name] = df
        systems.append((name, role))
    if not frames:
        raise SystemExit(f"no per_utterance.csv under {eval_dir}")

    records = []
    for name, role in systems:
        df = frames[name]
        timeline = df["timeline"].iloc[0]
        for severity, part in by_severity(df):
            for metric in METRICS:
                n, mean, low, high = cluster_bootstrap(part[metric], part["client_id"], args.n_boot, args.seed)
                records.append({"system": name, "role": role, "timeline": timeline, "severity": severity,
                                "metric": metric, "n": n, "mean": mean, "ci_low": low, "ci_high": high})
    summary = pd.DataFrame(records)
    summary.to_csv(out / "summary_by_severity.csv", index=False)

    # Speaker similarity, with clean-vs-clean references.
    speaker = summary[summary["metric"].isin(["spk_sim_same", "spk_sim_other"])]
    speaker = speaker.pivot_table(index=["system", "role", "severity"], columns="metric",
                                  values=["mean", "ci_low", "ci_high"]).reset_index()
    speaker.columns = ["_".join(c for c in col if c and c != "metric").strip("_") for col in speaker.columns]
    reference_rows = []
    ecapa = eval_dir / "ecapa_clean_test.npz"
    if ecapa.exists():
        cached = np.load(ecapa, allow_pickle=False)
        emb, spk = cached["embedding"], cached["client_id"]
        sim = emb @ emb.T
        same = spk[:, None] == spk[None, :]
        np.fill_diagonal(same, False)
        other = spk[:, None] != spk[None, :]
        per_clip_same = np.array([sim[i, same[i]].mean() if same[i].any() else np.nan for i in range(len(spk))])
        per_clip_diff = np.array([sim[i, other[i]].mean() for i in range(len(spk))])
        for label, values in (("clean: other clip, same speaker", per_clip_same),
                              ("clean: clips of other speakers", per_clip_diff)):
            n, mean, low, high = cluster_bootstrap(values, spk, args.n_boot, args.seed)
            reference_rows.append({"system": label, "role": "reference", "severity": "clean",
                                   "mean_spk_sim_other": mean, "ci_low_spk_sim_other": low,
                                   "ci_high_spk_sim_other": high})
    speaker = pd.concat([speaker, pd.DataFrame(reference_rows)], ignore_index=True)
    speaker.to_csv(out / "speaker_similarity.csv", index=False)

    # Paired comparisons B vs A per clip.
    paired_tables = []
    for b_name, a_name in args.pair or PAIRS:
        if b_name not in frames or a_name not in frames:
            print(f"paired comparison skipped: need both {b_name} and {a_name}")
            continue
        a = frames[a_name].set_index(["utterance_id", "severity"])
        b = frames[b_name].set_index(["utterance_id", "severity"])
        per_clip = pd.DataFrame({"client_id": a["client_id"]}, index=a.index)
        for metric in PAIRED_METRICS:
            per_clip[f"{metric}_{a_name}"] = a[metric]
            per_clip[f"{metric}_{b_name}"] = b.loc[a.index, metric]
            per_clip[f"{metric}_diff"] = per_clip[f"{metric}_{b_name}"] - per_clip[f"{metric}_{a_name}"]
        per_clip = per_clip.reset_index()
        per_clip.to_csv(out / f"paired_{b_name}_vs_{a_name}_per_clip.csv", index=False)
        paired_rows = []
        for severity, part in by_severity(per_clip):
            for metric in PAIRED_METRICS:
                diff = part[f"{metric}_diff"]
                n, mean, low, high = cluster_bootstrap(diff, part["client_id"], args.n_boot, args.seed)
                valid = part[np.isfinite(diff)]
                d = valid[f"{metric}_diff"]
                better = (d < 0) if metric in LOWER_IS_BETTER else (d > 0)
                speaker_means = valid.groupby("client_id")[f"{metric}_diff"].mean()
                paired_rows.append({
                    "severity": severity, "metric": metric,
                    "better": "lower" if metric in LOWER_IS_BETTER else "higher", "n": n,
                    "mean_a": float(valid[f"{metric}_{a_name}"].mean()) if n else np.nan,
                    "mean_b": float(valid[f"{metric}_{b_name}"].mean()) if n else np.nan,
                    "mean_diff": mean, "ci_low": low, "ci_high": high,
                    "share_b_better": float(better.mean()) if n else np.nan,
                    "n_speakers": int(len(speaker_means)),
                    "p_wilcoxon_speaker": safe_wilcoxon(speaker_means),
                    "p_wilcoxon_clip": safe_wilcoxon(d),
                })
        paired = pd.DataFrame(paired_rows)
        paired.insert(0, "a", a_name)
        paired.insert(0, "b", b_name)
        paired.to_csv(out / f"paired_{b_name}_vs_{a_name}.csv", index=False)
        paired_tables.append((b_name, a_name, paired_rows))

    # Markdown report.
    infos = {name: json.loads((eval_dir / name / "run_info.json").read_text())
             for name, _ in systems if (eval_dir / name / "run_info.json").exists()}
    lines = [f"# CV 27.0 test evaluation ({len(keys)} rows per system)", "",
             "Mean [95% CI], CI = percentile bootstrap over speakers (client_id), "
             f"{args.n_boot} resamples, seed {args.seed}.", "", "## Systems and timelines", "",
             "| system | role | timeline | inference | checkpoint |", "|---|---|---|---|---|"]
    for name, role in systems:
        info = infos.get(name, {})
        ckpt = info.get("checkpoint") or {}
        ckpt_text = (f"epoch {ckpt.get('epoch')}, best_epoch {ckpt.get('best_epoch')}, "
                     f"sha256 {str(ckpt.get('sha256'))[:12]}") if ckpt else "–"
        lines.append(f"| {name} | {role} | {info.get('timeline', '?')} | {info.get('inference', '?')} | {ckpt_text} |")
    for title, metrics in GROUPS.items():
        lines += ["", f"## {title}", "", markdown_table(summary, metrics, systems)]
    if reference_rows:
        lines += ["", "Clean references for spk_sim_other: " + "; ".join(
            f"{r['system']} {fmt(r['mean_spk_sim_other'], r['ci_low_spk_sim_other'], r['ci_high_spk_sim_other'])}"
            for r in reference_rows)]
    for b_name, a_name, paired_rows in paired_tables:
        lines += ["", f"## Paired: {b_name} − {a_name} (per clip)", "",
                  f"| severity | metric | better | mean diff [95% CI] | share {b_name} better | "
                  "p (Wilcoxon, speakers) | p (Wilcoxon, clips) |", "|---|---|---|---|---|---|---|"]
        for r in paired_rows:
            lines.append(f"| {r['severity']} | {r['metric']} | {r['better']} | "
                         f"{fmt(r['mean_diff'], r['ci_low'], r['ci_high'], 4)} | "
                         f"{r['share_b_better']:.2f} | {r['p_wilcoxon_speaker']:.3g} | "
                         f"{r['p_wilcoxon_clip']:.3g} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

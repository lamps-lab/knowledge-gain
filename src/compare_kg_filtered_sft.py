#!/usr/bin/env python3

import argparse
import json
import random
import statistics
from math import sqrt
from typing import Dict, List, Tuple, Optional

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


KG_KEYS = ["kg", "KG", "knowledge_gain", "mean_kg"]
PRE_KEYS = ["pre", "pre_acc", "pre_accuracy", "mean_pre"]
POST_KEYS = ["post", "post_acc", "post_accuracy", "mean_post"]


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_metric(row: dict, keys: List[str], required: bool = True) -> Optional[float]:
    for key in keys:
        if key in row and row[key] is not None:
            return float(row[key])

    if required:
        raise KeyError(f"Could not find any of keys {keys} in row keys: {list(row.keys())}")

    return None


def mean(xs: List[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def std(xs: List[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else float("nan")


def fmt(x: float, digits: int = 4) -> str:
    if x != x:
        return "--"
    return f"{x:.{digits}f}"


def fmt_pct(x: float, digits: int = 1) -> str:
    if x != x:
        return "--"
    return f"{x:.{digits}f}\\%"


def fmt_p(x: float) -> str:
    if x != x:
        return "--"
    if x < 1e-4:
        return f"{x:.2e}"
    return f"{x:.4f}"


def bootstrap_ci(
    values: List[float],
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    boot_means = []

    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(mean(sample))

    boot_means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot)
    hi_idx = min(hi_idx, n_boot - 1)

    return boot_means[lo_idx], boot_means[hi_idx]


def index_rows(rows: List[dict], id_key: Optional[str]) -> Dict[str, dict]:
    if id_key is None:
        return {str(i): row for i, row in enumerate(rows)}

    indexed = {}
    for row in rows:
        if id_key not in row:
            raise KeyError(f"ID key '{id_key}' not found in row keys: {list(row.keys())}")
        indexed[str(row[id_key])] = row

    return indexed


def paired_rows(
    base_rows: List[dict],
    model_rows: List[dict],
    id_key: Optional[str],
) -> Tuple[List[str], List[dict], List[dict]]:
    base_by_id = index_rows(base_rows, id_key)
    model_by_id = index_rows(model_rows, id_key)

    common_ids = sorted(set(base_by_id) & set(model_by_id))

    if not common_ids:
        raise ValueError("No paired examples found. Check --id-key or file contents.")

    return (
        common_ids,
        [base_by_id[i] for i in common_ids],
        [model_by_id[i] for i in common_ids],
    )


def summarize_model(rows: List[dict]) -> Dict[str, float]:
    pre = [get_metric(r, PRE_KEYS) for r in rows]
    post = [get_metric(r, POST_KEYS) for r in rows]
    kg = [get_metric(r, KG_KEYS) for r in rows]

    return {
        "pre": mean(pre),
        "post": mean(post),
        "kg": mean(kg),
    }


def paired_stats(base_rows: List[dict], model_rows: List[dict]) -> Dict[str, float]:
    base_kg = [get_metric(r, KG_KEYS) for r in base_rows]
    model_kg = [get_metric(r, KG_KEYS) for r in model_rows]

    diffs = [m - b for b, m in zip(base_kg, model_kg)]

    base_mean = mean(base_kg)
    model_mean = mean(model_kg)
    diff_mean = mean(diffs)
    diff_median = statistics.median(diffs)
    diff_sd = std(diffs)
    diff_se = diff_sd / sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
    rel_improvement = (diff_mean / base_mean * 100) if base_mean != 0 else float("nan")

    ci_lo, ci_hi = bootstrap_ci(diffs)

    wins = sum(d > 0 for d in diffs)
    ties = sum(d == 0 for d in diffs)
    losses = sum(d < 0 for d in diffs)

    cohen_dz = diff_mean / diff_sd if diff_sd and diff_sd > 0 else float("nan")

    wilcoxon_stat = float("nan")
    wilcoxon_two_sided_p = float("nan")
    wilcoxon_greater_p = float("nan")

    if wilcoxon is not None:
        try:
            stat_two, p_two = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
            stat_greater, p_greater = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
            wilcoxon_stat = float(stat_two)
            wilcoxon_two_sided_p = float(p_two)
            wilcoxon_greater_p = float(p_greater)
        except ValueError:
            # Happens if all differences are zero.
            pass

    return {
        "base_kg": base_mean,
        "model_kg": model_mean,
        "mean_diff": diff_mean,
        "median_diff": diff_median,
        "relative_improvement_pct": rel_improvement,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "diff_sd": diff_sd,
        "diff_se": diff_se,
        "cohen_dz": cohen_dz,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_two_sided_p": wilcoxon_two_sided_p,
        "wilcoxon_greater_p": wilcoxon_greater_p,
    }


def parse_model_arg(s: str) -> Tuple[str, str, str]:
    """
    Format:
      NAME=FILTER=PATH

    Example:
      SFT-KG-Top75=Top 75\\% by KG=results/eval_sft_kg_top75_s5.jsonl
    """
    parts = s.split("=", 2)
    if len(parts) != 3:
        raise ValueError(
            "Each --model must use NAME=FILTER=PATH, e.g. "
            "'SFT-KG-Top75=Top 75\\% by KG=results/top75.jsonl'"
        )
    return parts[0], parts[1], parts[2]


def latex_escape(s: str) -> str:
    return (
        s.replace("&", r"\&")
        .replace("_", r"\_")
        .replace("%", r"\%")
    )


def make_latex_table(
    base_name: str,
    base_summary: Dict[str, float],
    rows: List[Dict[str, object]],
    include_stats_cols: bool = False,
) -> str:
    if include_stats_cols:
        header = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrrr}
\toprule
Model & Filter & Pre & Post & KG & $\Delta$KG & Rel. $\Delta$ & $p$ \\
\midrule
"""
    else:
        header = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrr}
\toprule
Model & Filter & Pre & Post & KG & $\Delta$KG \\
\midrule
"""

    if include_stats_cols:
        body = (
            f"{latex_escape(base_name)} & All "
            f"& {fmt(base_summary['pre'])} "
            f"& {fmt(base_summary['post'])} "
            f"& {fmt(base_summary['kg'])} "
            f"& -- & -- & -- \\\\\n"
        )
    else:
        body = (
            f"{latex_escape(base_name)} & All "
            f"& {fmt(base_summary['pre'])} "
            f"& {fmt(base_summary['post'])} "
            f"& {fmt(base_summary['kg'])} "
            f"& -- \\\\\n"
        )

    for r in rows:
        name = latex_escape(str(r["name"]))
        filt = str(r["filter"])
        summary = r["summary"]
        stats = r["stats"]

        kg_str = fmt(summary["kg"])
        diff_str = fmt(stats["mean_diff"])

        if stats["mean_diff"] == max(x["stats"]["mean_diff"] for x in rows):
            kg_str = r"\textbf{" + kg_str + "}"
            diff_str = r"\textbf{+" + diff_str + "}"
        else:
            diff_str = "+" + diff_str if stats["mean_diff"] >= 0 else diff_str

        if include_stats_cols:
            body += (
                f"{name} & {filt} "
                f"& {fmt(summary['pre'])} "
                f"& {fmt(summary['post'])} "
                f"& {kg_str} "
                f"& {diff_str} "
                f"& {fmt_pct(stats['relative_improvement_pct'])} "
                f"& {fmt_p(stats['wilcoxon_two_sided_p'])} \\\\\n"
            )
        else:
            body += (
                f"{name} & {filt} "
                f"& {fmt(summary['pre'])} "
                f"& {fmt(summary['post'])} "
                f"& {kg_str} "
                f"& {diff_str} \\\\\n"
            )

    footer = r"""\bottomrule
\end{tabular}
\caption{Held-out LLMSim evaluation on 300 examples with five simulated readers per article. KG-filtered SFT improves post-reading accuracy while leaving pre-reading accuracy nearly unchanged.}
\label{tab:kg_filtered_sft}
\end{table}
"""
    if include_stats_cols:
        footer = footer.replace(r"\end{table}", r"\end{table*}")

    return header + body + footer


def print_text_summary(base_name: str, rows: List[Dict[str, object]]) -> None:
    print("\nPaper-ready result sentences:\n")

    best = max(rows, key=lambda r: r["stats"]["mean_diff"])
    bstats = best["stats"]

    print(
        f"{best['name']} achieves the strongest held-out simulated "
        f"{{\\sc KnowledgeGain}}, improving mean KG from "
        f"{fmt(bstats['base_kg'])} to {fmt(bstats['model_kg'])}. "
        f"This corresponds to a {fmt(bstats['relative_improvement_pct'], 1)}\\% "
        f"relative improvement over {base_name}. "
        f"The paired mean difference is {fmt(bstats['mean_diff'])} "
        f"with a bootstrap 95\\% CI of "
        f"[{fmt(bstats['bootstrap_ci_lo'])}, {fmt(bstats['bootstrap_ci_hi'])}], "
        f"and a two-sided Wilcoxon signed-rank test gives "
        f"$p={fmt_p(bstats['wilcoxon_two_sided_p'])}$."
    )

    print("\nDetailed paired comparisons:")
    for r in rows:
        stats = r["stats"]
        print(
            f"- {r['name']}: ΔKG={fmt(stats['mean_diff'])}, "
            f"rel={fmt(stats['relative_improvement_pct'], 1)}%, "
            f"CI=[{fmt(stats['bootstrap_ci_lo'])}, {fmt(stats['bootstrap_ci_hi'])}], "
            f"Wilcoxon two-sided p={fmt_p(stats['wilcoxon_two_sided_p'])}, "
            f"wins/ties/losses={stats['wins']}/{stats['ties']}/{stats['losses']}, "
            f"dz={fmt(stats['cohen_dz'])}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        required=True,
        help="Baseline SFT-All evaluation JSONL.",
    )
    ap.add_argument(
        "--base-name",
        default="SFT-All",
        help="Name for the baseline model.",
    )
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Model comparison in NAME=FILTER=PATH format. "
            "Can be repeated."
        ),
    )
    ap.add_argument(
        "--id-key",
        default=None,
        help="Stable example ID key. If omitted, rows are paired by line number.",
    )
    ap.add_argument(
        "--stats-table",
        action="store_true",
        help="Include relative improvement and p-values in the LaTeX table.",
    )
    args = ap.parse_args()

    base_rows_raw = load_jsonl(args.base)
    model_specs = [parse_model_arg(x) for x in args.model]

    all_results = []

    # For baseline summary, use all rows from the first model pairing if possible.
    # This ensures the baseline summary is computed over the same paired examples.
    first_model_rows = load_jsonl(model_specs[0][2])
    _, base_rows_first, _ = paired_rows(base_rows_raw, first_model_rows, args.id_key)
    base_summary = summarize_model(base_rows_first)

    print("=" * 80)
    print("KG-filtered SFT multi-model comparison")
    print("=" * 80)
    print(f"Baseline: {args.base_name}")
    print(f"Base file: {args.base}")
    print(f"Pairing: {'line number' if args.id_key is None else args.id_key}")

    print(
        f"\n{args.base_name}: "
        f"Pre={fmt(base_summary['pre'])}, "
        f"Post={fmt(base_summary['post'])}, "
        f"KG={fmt(base_summary['kg'])}"
    )

    for name, filt, path in model_specs:
        model_rows_raw = load_jsonl(path)
        common_ids, base_rows, model_rows = paired_rows(base_rows_raw, model_rows_raw, args.id_key)

        base_summary_for_model = summarize_model(base_rows)
        model_summary = summarize_model(model_rows)
        stats = paired_stats(base_rows, model_rows)

        all_results.append({
            "name": name,
            "filter": filt,
            "path": path,
            "n": len(common_ids),
            "base_summary": base_summary_for_model,
            "summary": model_summary,
            "stats": stats,
        })

        print(f"\n{name}")
        print(f"  file: {path}")
        print(f"  N paired: {len(common_ids)}")
        print(
            f"  Pre={fmt(model_summary['pre'])}, "
            f"Post={fmt(model_summary['post'])}, "
            f"KG={fmt(model_summary['kg'])}"
        )
        print(
            f"  ΔKG={fmt(stats['mean_diff'])}, "
            f"relative={fmt(stats['relative_improvement_pct'], 2)}%, "
            f"CI=[{fmt(stats['bootstrap_ci_lo'])}, {fmt(stats['bootstrap_ci_hi'])}], "
            f"p(two-sided)={fmt_p(stats['wilcoxon_two_sided_p'])}, "
            f"p(greater)={fmt_p(stats['wilcoxon_greater_p'])}, "
            f"wins/ties/losses={stats['wins']}/{stats['ties']}/{stats['losses']}"
        )

    print("\n" + "=" * 80)
    print("LaTeX table")
    print("=" * 80)
    print(make_latex_table(args.base_name, base_summary, all_results, include_stats_cols=args.stats_table))

    print_text_summary(args.base_name, all_results)


if __name__ == "__main__":
    main()
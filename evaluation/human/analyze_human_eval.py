#!/usr/bin/env python3
"""
python3 analyze_human_eval.py \
  --pointwise pointwise.csv \
  --pairwise pairwise.csv \
  --out_dir human_eval_analysis

OR

python3 analyze_human_eval.py --pointwise human/pointwise.csv --pairwise human/pairwise.csv --judge results/open-source/pointwise_top50.json --pairwise_judge llm_judge_pairwise_news0_vs_news2_claude_sonnet.json --corr pearson --out_dir human_eval_analysis
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Pointwise survey positions 1..20. These are the positions used by the Qualtrics
# export columns, not necessarily the underlying article IDs.
POINTWISE_ARTICLE_IDS = [
    29733, 29734, 29739, 29753, 29775,
    29802, 29846, 29866, 29873, 29929,
    29902, 29943, 29958, 29988, 30052,
    30061, 30086, 30184, 30214, 30708,
]

POINTWISE_OURS_POSITIONS = {1, 4, 6, 7, 9, 12, 15, 16, 18, 19}

# Pairwise A/B randomization.
# pair_index: (article_id, article_A_system, article_B_system)
PAIRWISE_RANDOMIZATION = {
    1: (29733, "news_0", "news_2"),
    2: (29734, "news_2", "news_0"),
    3: (29739, "news_0", "news_2"),
    4: (29753, "news_0", "news_2"),
    5: (29775, "news_2", "news_0"),
    6: (29802, "news_0", "news_2"),
    7: (29846, "news_0", "news_2"),
    8: (29866, "news_2", "news_0"),
    9: (29873, "news_2", "news_0"),
    10: (29929, "news_2", "news_0"),
    11: (29902, "news_0", "news_2"),
    12: (29943, "news_0", "news_2"),
    13: (29958, "news_0", "news_2"),
    14: (29988, "news_2", "news_0"),
    15: (30052, "news_0", "news_2"),
    16: (30061, "news_2", "news_0"),
    17: (30086, "news_2", "news_0"),
    18: (30184, "news_0", "news_2"),
    19: (30214, "news_2", "news_0"),
    20: (30708, "news_0", "news_2"),
}

REASON_LABELS = {
    1: "Accuracy",
    2: "Completeness",
    3: "Relevance",
    4: "Clarity",
}

DIMENSIONS = ["Accuracy", "Completeness", "Relevance", "Clarity"]


def read_json_or_jsonl(path: str) -> Any:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".jsonl":
        rows = []
        with open(path_obj, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path_obj, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_out_dir(path: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def recode_pointwise_score(x: Any) -> float:
    """Qualtrics option 1 is best and option 5 is worst; convert to 5=best, 1=worst."""
    v = safe_float(x)
    if np.isnan(v):
        return np.nan
    return 6.0 - v


def fmt_mean_ci(mean: float, ci: float) -> str:
    if np.isnan(mean):
        return "--"
    if np.isnan(ci):
        return f"{mean:.2f}"
    return f"${mean:.2f}{{\\pm}}{ci:.2f}$"


def fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "--"
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}"


def fmt_pct(x: float) -> str:
    if x is None or np.isnan(x):
        return "--"
    return f"{100*x:.1f}"


def t_critical_975(df: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        return table.get(df, 1.96)


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return center - half, center + half


def exact_binom_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial test for H0 p=p0. Uses scipy if available."""
    if n == 0:
        return np.nan
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, p=p0, alternative="two-sided").pvalue)
    except Exception:
        # Exact p-value by probability ordering. Fine for n=100.
        probs = []
        obs_prob = math.comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k))
        for i in range(n + 1):
            pr = math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i))
            if pr <= obs_prob + 1e-15:
                probs.append(pr)
        return min(1.0, sum(probs))


def paired_t_pvalue(diffs: pd.Series) -> float:
    diffs = pd.to_numeric(diffs, errors="coerce").dropna()
    if len(diffs) < 2 or diffs.nunique() < 2:
        return np.nan
    try:
        from scipy.stats import ttest_1samp
        return float(ttest_1samp(diffs, popmean=0.0).pvalue)
    except Exception:
        return np.nan


def corr_pair(df: pd.DataFrame, x: str, y: str, method: str) -> float:
    if x not in df.columns or y not in df.columns:
        return np.nan
    sub = df[[x, y]].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 2 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
        return np.nan
    return float(sub[x].corr(sub[y], method=method))


def load_pointwise(pointwise_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses the pointwise Qualtrics CSV.

    The first two rows are Qualtrics metadata rows, so data starts at row index 2.
    Each displayed article takes 8 columns: rating/reasoning for Accuracy,
    Completeness, Relevance, and Clarity.
    """
    df = pd.read_csv(pointwise_path, low_memory=False)

    records = []
    reason_records = []

    for row_idx in range(2, len(df)):
        row = df.iloc[row_idx]
        respondent = str(row.get("ResponseId", f"row_{row_idx}"))

        for position in range(1, 21):
            article_id = POINTWISE_ARTICLE_IDS[position - 1]
            system = "Ours" if position in POINTWISE_OURS_POSITIONS else "Agentic"
            system_key = "news_0" if system == "Ours" else "news_2"

            start_idx = 17 + 8 * (position - 1)
            rating_indices = [start_idx, start_idx + 2, start_idx + 4, start_idx + 6]
            reason_indices = [start_idx + 1, start_idx + 3, start_idx + 5, start_idx + 7]

            for dim, rating_idx, reason_idx in zip(DIMENSIONS, rating_indices, reason_indices):
                score = recode_pointwise_score(row.iloc[rating_idx])
                reason_text = row.iloc[reason_idx]
                if isinstance(reason_text, float) and np.isnan(reason_text):
                    reason_text = ""
                else:
                    reason_text = str(reason_text)

                records.append(
                    {
                        "respondent": respondent,
                        "position": position,
                        "article_id": article_id,
                        "system": system,
                        "system_key": system_key,
                        "dimension": dim,
                        "score": score,
                    }
                )

                reason_records.append(
                    {
                        "respondent": respondent,
                        "position": position,
                        "article_id": article_id,
                        "system": system,
                        "system_key": system_key,
                        "dimension": dim,
                        "score": score,
                        "reason_text": reason_text,
                    }
                )

    return pd.DataFrame(records), pd.DataFrame(reason_records)


def summarize_pointwise(pointwise_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Respondent-level means are used for confidence intervals, because each respondent rates multiple articles.
    respondent_means = (
        pointwise_long.groupby(["respondent", "system", "dimension"], as_index=False)["score"]
        .mean()
    )

    summary = (
        respondent_means.groupby(["system", "dimension"], as_index=False)["score"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary["tcrit"] = summary["count"].apply(lambda n: t_critical_975(int(n) - 1) if n > 1 else np.nan)
    summary["ci95"] = summary["tcrit"] * summary["std"] / np.sqrt(summary["count"])

    # Add paired p-values based on respondent-level Ours-Agentic differences.
    pivot = respondent_means.pivot_table(
        index=["respondent", "dimension"],
        columns="system",
        values="score",
    ).reset_index()
    if "Ours" in pivot.columns and "Agentic" in pivot.columns:
        pivot["diff"] = pivot["Ours"] - pivot["Agentic"]
        pvals = pivot.groupby("dimension")["diff"].apply(paired_t_pvalue).rename("paired_p").reset_index()
        summary = summary.merge(pvals, on="dimension", how="left")
    else:
        summary["paired_p"] = np.nan

    # Overall score across all four dimensions.
    overall_resp = pointwise_long.groupby(["respondent", "system"], as_index=False)["score"].mean()
    overall = overall_resp.groupby("system", as_index=False)["score"].agg(["count", "mean", "std"]).reset_index()
    overall["tcrit"] = overall["count"].apply(lambda n: t_critical_975(int(n) - 1) if n > 1 else np.nan)
    overall["ci95"] = overall["tcrit"] * overall["std"] / np.sqrt(overall["count"])

    if set(["Ours", "Agentic"]).issubset(set(overall_resp["system"])):
        opiv = overall_resp.pivot(index="respondent", columns="system", values="score")
        if "Ours" in opiv.columns and "Agentic" in opiv.columns:
            overall_p = paired_t_pvalue(opiv["Ours"] - opiv["Agentic"])
        else:
            overall_p = np.nan
    else:
        overall_p = np.nan
    overall["paired_p"] = overall_p
    overall["dimension"] = "Mean"

    summary_with_mean = pd.concat([summary, overall], ignore_index=True, sort=False)
    return summary_with_mean, respondent_means


def make_pointwise_latex(summary: pd.DataFrame) -> str:
    order_systems = ["Ours", "Agentic"]
    order_dims = ["Accuracy", "Completeness", "Relevance", "Clarity", "Mean"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"System & Accuracy & Completeness & Relevance & Clarity & Mean \\")
    lines.append(r"\midrule")

    for system in order_systems:
        cells = []
        for dim in order_dims:
            row = summary[(summary["system"] == system) & (summary["dimension"] == dim)]
            if row.empty:
                cells.append("--")
            else:
                r = row.iloc[0]
                cells.append(fmt_mean_ci(float(r["mean"]), float(r["ci95"])))
        lines.append(f"{system} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Human pointwise evaluation of generated articles. Scores are on a 1--5 scale, with higher values indicating better quality. Values are means with 95\% confidence intervals computed over respondent-level means.}"
    )
    lines.append(r"\label{tab:human_pointwise_generated}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def load_pairwise(pairwise_path: str) -> pd.DataFrame:
    df = pd.read_csv(pairwise_path, low_memory=False)

    choice_cols = []
    for i, col in enumerate(df.columns):
        if i >= 17 and str(df.iloc[0, i]).startswith("Which news article is better overall"):
            choice_cols.append((i, col))

    if len(choice_cols) != 20:
        print(f"Warning: expected 20 pairwise choice columns, found {len(choice_cols)}.")

    records = []

    for row_idx in range(2, len(df)):
        row = df.iloc[row_idx]
        respondent = str(row.get("ResponseId", f"row_{row_idx}"))

        for pair_index, (choice_idx, choice_col) in enumerate(choice_cols, start=1):
            if pair_index not in PAIRWISE_RANDOMIZATION:
                continue

            choice_val = safe_float(row.iloc[choice_idx])
            if np.isnan(choice_val):
                continue
            choice = int(choice_val)
            if choice not in {1, 2}:
                continue

            article_id, system_a, system_b = PAIRWISE_RANDOMIZATION[pair_index]
            chosen_system_key = system_a if choice == 1 else system_b
            chosen_system = "Ours" if chosen_system_key == "news_0" else "Agentic"

            reason_code = np.nan
            reason_label = ""
            # Qualtrics export has a reason column immediately after the choice for pairs 2..20.
            if choice_idx + 1 < df.shape[1] and str(df.iloc[0, choice_idx + 1]).startswith("Please select the reason"):
                reason_val = safe_float(row.iloc[choice_idx + 1])
                if not np.isnan(reason_val):
                    reason_int = int(reason_val)
                    if reason_int in REASON_LABELS:
                        reason_code = reason_int
                        reason_label = REASON_LABELS[reason_int]

            records.append(
                {
                    "respondent": respondent,
                    "pair_index": pair_index,
                    "article_id": article_id,
                    "system_a": system_a,
                    "system_b": system_b,
                    "choice": choice,
                    "chosen_system_key": chosen_system_key,
                    "chosen_system": chosen_system,
                    "ours_win": 1 if chosen_system_key == "news_0" else 0,
                    "reason_code": reason_code,
                    "reason_label": reason_label,
                }
            )

    return pd.DataFrame(records)


def summarize_pairwise(pairwise_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = int(pairwise_long["ours_win"].notna().sum())
    k = int(pairwise_long["ours_win"].sum())
    win_rate = k / n if n else np.nan
    lo, hi = wilson_ci(k, n)
    p = exact_binom_two_sided(k, n, p0=0.5)

    summary = pd.DataFrame(
        [
            {
                "pair": "Ours vs Agentic",
                "wins": k,
                "n": n,
                "win_rate": win_rate,
                "ci95_low": lo,
                "ci95_high": hi,
                "p_value": p,
            }
        ]
    )

    reason_df = pairwise_long.dropna(subset=["reason_code"]).copy()
    if reason_df.empty:
        reason_summary = pd.DataFrame(columns=["reason", "count", "percent"])
    else:
        reason_summary = (
            reason_df.groupby("reason_label", as_index=False)
            .size()
            .rename(columns={"reason_label": "reason", "size": "count"})
        )
        reason_summary["percent"] = reason_summary["count"] / reason_summary["count"].sum()
        reason_summary = reason_summary.sort_values("count", ascending=False)

    return summary, reason_summary


def make_pairwise_latex(summary: pd.DataFrame) -> str:
    r = summary.iloc[0]
    ci = f"[{fmt_pct(float(r['ci95_low']))}, {fmt_pct(float(r['ci95_high']))}]"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Pair & Win rate for Ours & 95\% CI & $p$-value \\")
    lines.append(r"\midrule")
    lines.append(f"{r['pair']} & {fmt_pct(float(r['win_rate']))} & {ci} & {fmt_p(float(r['p_value']))} " + "\\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Human pairwise preference between our KG-optimized system and the agentic baseline. The confidence interval is Wilson's 95\% interval and the $p$-value is from an exact two-sided binomial test against chance preference.}"
    )
    lines.append(r"\label{tab:human_pairwise_generated}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_reason_latex(reason_summary: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Reason & Count & Share \\")
    lines.append(r"\midrule")
    for _, r in reason_summary.iterrows():
        lines.append(f"{r['reason']} & {int(r['count'])} & {fmt_pct(float(r['percent']))} " + "\\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Reasons selected by annotators for pairwise preferences. Invalid or missing reason codes are excluded from this summary.}"
    )
    lines.append(r"\label{tab:human_pairwise_reasons}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def load_llm_judge(judge_path: str) -> pd.DataFrame:
    data = read_json_or_jsonl(judge_path)
    rows = []
    for item in data:
        article_id = int(item["id"])
        for system_key, scores in item.get("scores_by_system", {}).items():
            if system_key not in {"news_0", "news_2"}:
                continue
            system = "Ours" if system_key == "news_0" else "Agentic"
            rows.append(
                {
                    "article_id": article_id,
                    "system_key": system_key,
                    "system": system,
                    "llm_accuracy": scores.get("accuracy"),
                    "llm_completeness": scores.get("completeness"),
                    "llm_relevance": scores.get("relevance"),
                    "llm_clarity": scores.get("clarity"),
                    "llm_expected_kgain": scores.get("knowledge_gain"),
                    "llm_mean_score": scores.get("mean_score"),
                }
            )
    return pd.DataFrame(rows)


def summarize_human_for_judge(pointwise_long: pd.DataFrame) -> pd.DataFrame:
    human_dim = (
        pointwise_long.groupby(["article_id", "system_key", "system", "dimension"], as_index=False)["score"]
        .mean()
    )
    wide = human_dim.pivot_table(
        index=["article_id", "system_key", "system"],
        columns="dimension",
        values="score",
    ).reset_index()
    wide.columns.name = None
    rename = {dim: f"human_{dim.lower()}" for dim in DIMENSIONS}
    wide = wide.rename(columns=rename)
    human_cols = [f"human_{dim.lower()}" for dim in DIMENSIONS]
    wide["human_mean_score"] = wide[human_cols].mean(axis=1)
    return wide


def make_judge_alignment(
    pointwise_long: pd.DataFrame,
    judge_path: str,
    method: str,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    human = summarize_human_for_judge(pointwise_long)
    judge = load_llm_judge(judge_path)
    merged = human.merge(judge, on=["article_id", "system_key", "system"], how="inner")

    rows = []
    specs = [
        ("LLM judge accuracy", "llm_accuracy", "human_accuracy", "Human accuracy"),
        ("LLM judge completeness", "llm_completeness", "human_completeness", "Human completeness"),
        ("LLM judge relevance", "llm_relevance", "human_relevance", "Human relevance"),
        ("LLM judge clarity", "llm_clarity", "human_clarity", "Human clarity"),
        ("LLM judge expected KGain", "llm_expected_kgain", "human_mean_score", "Human mean pointwise score"),
        ("LLM judge mean score", "llm_mean_score", "human_mean_score", "Human mean pointwise score"),
    ]
    for label, x, y, target in specs:
        rows.append(
            {
                "Metric": label,
                "Human target": target,
                f"{method} corr.": corr_pair(merged, x, y, method),
                "n": int(merged[[x, y]].dropna().shape[0]) if x in merged.columns and y in merged.columns else 0,
            }
        )

    # Overall stacked correlation across the four matched dimensions.
    stacked = []
    for dim in ["accuracy", "completeness", "relevance", "clarity"]:
        for _, r in merged.iterrows():
            hv = r.get(f"human_{dim}")
            lv = r.get(f"llm_{dim}")
            if pd.notna(hv) and pd.notna(lv):
                stacked.append({"human": hv, "llm": lv, "dimension": dim})
    stacked_df = pd.DataFrame(stacked)
    overall_corr = corr_pair(stacked_df, "llm", "human", method) if not stacked_df.empty else np.nan

    alignment = pd.DataFrame(rows)
    alignment.loc[len(alignment)] = {
        "Metric": "Overall stacked dimensions",
        "Human target": "Matched human dimension",
        f"{method} corr.": overall_corr,
        "n": int(len(stacked_df)),
    }

    merged.to_csv(out_dir / "human_llm_judge_merged_points.csv", index=False)
    alignment.to_csv(out_dir / "human_llm_judge_alignment.csv", index=False)

    # Optional scatter figure.
    try:
        import matplotlib.pyplot as plt
        if not stacked_df.empty:
            plt.figure(figsize=(5, 4))
            plt.scatter(stacked_df["llm"], stacked_df["human"], alpha=0.8)
            plt.xlabel("LLM judge score")
            plt.ylabel("Human pointwise score")
            plt.title(f"Human--LLM pointwise alignment ({method}={overall_corr:.2f})")
            plt.tight_layout()
            plt.savefig(out_dir / "human_llm_correlation.pdf")
            plt.close()
    except Exception as e:
        print(f"Warning: could not create LLM-human correlation plot: {e}")

    return alignment, merged

def make_judge_alignment_latex(alignment: pd.DataFrame, method: str) -> str:
    corr_col = f"{method} corr."
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(rf"Metric & {method.title()} corr. & $n$ \\")
    lines.append(r"\midrule")
    for _, r in alignment.iterrows():
        val = r[corr_col]
        val_str = "--" if pd.isna(val) else f"{float(val):.3f}"
        lines.append(f"{r['Metric']} & {val_str} & {int(r['n'])} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Alignment between LLM judge scores and human pointwise ratings. For accuracy, completeness, relevance, and clarity, each LLM dimension is correlated with the matching human dimension. Expected KGain is correlated with the mean human pointwise score because the human pointwise survey did not include an expected-KGain rating.}"
    )
    lines.append(r"\label{tab:human_llm_judge_alignment}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def corr_with_pvalue(df: pd.DataFrame, x: str, y: str, method: str) -> Tuple[float, float, int]:
    """Correlation with p-value for Pearson/Spearman/Kendall."""
    if x not in df.columns or y not in df.columns:
        return np.nan, np.nan, 0

    sub = df[[x, y]].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna()

    n = len(sub)
    if n < 2 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
        return np.nan, np.nan, n

    try:
        if method == "pearson":
            from scipy.stats import pearsonr
            r, p = pearsonr(sub[x], sub[y])
        elif method == "spearman":
            from scipy.stats import spearmanr
            r, p = spearmanr(sub[x], sub[y])
        elif method == "kendall":
            from scipy.stats import kendalltau
            r, p = kendalltau(sub[x], sub[y])
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        return float(r), float(p), n
    except Exception:
        return float(sub[x].corr(sub[y], method=method)), np.nan, n


def load_pairwise_llm_judge(pairwise_judge_path: str) -> pd.DataFrame:
    """
    Load LLM pairwise judge file.

    Expected format:
      [
        {
          "id": 29733,
          "winner_key": "news_0" or "news_2",
          "judgment": {"reason": "..."}
        },
        ...
      ]

    We score from the Ours/news_0 perspective:
      news_0 -> 1.0
      news_2 -> 0.0
      explicit tie -> 0.5
    """
    data = read_json_or_jsonl(pairwise_judge_path)

    rows = []
    for rec in data:
        article_id = int(rec.get("id"))
        winner_key = rec.get("winner_key")
        winner_label = rec.get("winner_label")

        if winner_key == "news_0":
            llm_ours_score = 1.0
            llm_choice = "Ours"
        elif winner_key == "news_2":
            llm_ours_score = 0.0
            llm_choice = "Agentic"
        elif winner_key in {"tie", "both", "equal"} or winner_label in {"tie", "both", "equal"}:
            llm_ours_score = 0.5
            llm_choice = "Tie"
        else:
            llm_ours_score = np.nan
            llm_choice = "Unknown"

        rows.append(
            {
                "article_id": article_id,
                "category": rec.get("category"),
                "llm_winner_key": winner_key,
                "llm_winner_label": winner_label,
                "llm_ours_score": llm_ours_score,
                "llm_choice": llm_choice,
                "llm_reason": (rec.get("judgment") or {}).get("reason"),
            }
        )

    return pd.DataFrame(rows)


def summarize_human_pairwise_by_article(pairwise_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate human pairwise choices to one row per article."""
    human = (
        pairwise_long.groupby("article_id", as_index=False)
        .agg(
            human_ours_win_rate=("ours_win", "mean"),
            n_human_votes=("ours_win", "count"),
        )
    )

    human["human_majority_score"] = np.where(
        human["human_ours_win_rate"] > 0.5,
        1.0,
        np.where(human["human_ours_win_rate"] < 0.5, 0.0, 0.5),
    )

    human["human_majority_choice"] = np.where(
        human["human_majority_score"] == 1.0,
        "Ours",
        np.where(human["human_majority_score"] == 0.0, "Agentic", "Tie"),
    )

    return human


def make_pairwise_judge_alignment(
    pairwise_long: pd.DataFrame,
    pairwise_judge_path: str,
    method: str,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare human pairwise preferences with LLM pairwise preferences.

    Main correlation:
      human_ours_win_rate vs. llm_ours_score

    Agreement:
      human majority choice vs. LLM choice
    """
    human = summarize_human_pairwise_by_article(pairwise_long)
    llm = load_pairwise_llm_judge(pairwise_judge_path)

    merged = human.merge(llm, on="article_id", how="inner")

    # Correlation between continuous human win rate and LLM binary/ternary score.
    corr, corr_p, n_corr = corr_with_pvalue(
        merged,
        "human_ours_win_rate",
        "llm_ours_score",
        method,
    )

    # Agreement between human majority and LLM winner.
    valid_agree = merged[
        merged["human_majority_choice"].isin(["Ours", "Agentic"])
        & merged["llm_choice"].isin(["Ours", "Agentic"])
    ].copy()

    if len(valid_agree) > 0:
        valid_agree["majority_agrees"] = (
            valid_agree["human_majority_choice"] == valid_agree["llm_choice"]
        ).astype(int)
        agreement = float(valid_agree["majority_agrees"].mean())
        n_agreement = int(len(valid_agree))
    else:
        agreement = np.nan
        n_agreement = 0

    # Directional counts.
    llm_ours_n = int((merged["llm_choice"] == "Ours").sum())
    llm_agentic_n = int((merged["llm_choice"] == "Agentic").sum())
    llm_tie_n = int((merged["llm_choice"] == "Tie").sum())

    human_ours_majority_n = int((merged["human_majority_choice"] == "Ours").sum())
    human_agentic_majority_n = int((merged["human_majority_choice"] == "Agentic").sum())
    human_tie_majority_n = int((merged["human_majority_choice"] == "Tie").sum())

    summary = pd.DataFrame(
        [
            {
                "comparison": "Human pairwise vs LLM pairwise",
                "method": method,
                "correlation": corr,
                "correlation_p": corr_p,
                "n_correlation_articles": n_corr,
                "majority_agreement": agreement,
                "n_agreement_articles": n_agreement,
                "human_ours_majority_n": human_ours_majority_n,
                "human_agentic_majority_n": human_agentic_majority_n,
                "human_tie_majority_n": human_tie_majority_n,
                "llm_ours_n": llm_ours_n,
                "llm_agentic_n": llm_agentic_n,
                "llm_tie_n": llm_tie_n,
            }
        ]
    )

    merged.to_csv(out_dir / "human_llm_pairwise_merged.csv", index=False)
    summary.to_csv(out_dir / "human_llm_pairwise_alignment.csv", index=False)

    # Scatter/crossplot.
    try:
        import matplotlib.pyplot as plt

        plot_df = merged.dropna(subset=["human_ours_win_rate", "llm_ours_score"]).copy()
        if not plot_df.empty:
            plt.figure(figsize=(5, 4))
            plt.scatter(
                plot_df["human_ours_win_rate"],
                plot_df["llm_ours_score"],
                alpha=0.85,
            )
            plt.xlabel("Human Ours win rate")
            plt.ylabel("LLM Ours score")
            plt.title(f"Human--LLM pairwise alignment\n{method}={corr:.2f}, p={corr_p:.3f}")
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            plt.savefig(out_dir / "human_llm_pairwise_correlation.pdf")
            plt.close()
    except Exception as e:
        print(f"Warning: could not create pairwise LLM-human plot: {e}")

    return summary, merged


def make_pairwise_judge_alignment_latex(summary: pd.DataFrame) -> str:
    r = summary.iloc[0]

    corr = r["correlation"]
    corr_p = r["correlation_p"]
    agreement = r["majority_agreement"]

    corr_str = "--" if pd.isna(corr) else f"{float(corr):.3f}"
    corr_p_str = fmt_p(float(corr_p)) if not pd.isna(corr_p) else "--"
    agreement_str = "--" if pd.isna(agreement) else fmt_pct(float(agreement))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Comparison & Corr. & $p$ & Majority agreement & $n$ \\")
    lines.append(r"\midrule")
    lines.append(
        f"Human vs. LLM pairwise "
        f"& {corr_str} "
        f"& {corr_p_str} "
        f"& {agreement_str} "
        f"& {int(r['n_correlation_articles'])} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Human--LLM pairwise alignment for generated article preferences. "
        r"The correlation compares article-level human Ours win rate with the LLM judge's Ours score "
        r"($1$ if the LLM selects Ours, $0$ if it selects Agentic). Majority agreement compares the "
        r"human majority choice for each article with the LLM pairwise choice.}"
    )
    lines.append(r"\label{tab:human_llm_pairwise_alignment}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise", required=True, help="Path to pointwise Qualtrics CSV.")
    parser.add_argument("--pairwise", required=True, help="Path to pairwise Qualtrics CSV.")
    parser.add_argument("--judge", default=None, help="Optional LLM judge JSON/JSONL file.")
    parser.add_argument("--pairwise_judge",default=None,help="Optional LLM pairwise judge JSON/JSONL file with winner_key per article.")
    parser.add_argument("--corr", choices=["pearson", "spearman", "kendall"], default="spearman")
    parser.add_argument("--out_dir", default="human_eval_analysis")
    args = parser.parse_args()

    out_dir = ensure_out_dir(args.out_dir)

    pointwise_long, pointwise_reasons = load_pointwise(args.pointwise)
    pointwise_summary, pointwise_respondent_means = summarize_pointwise(pointwise_long)

    pointwise_long.to_csv(out_dir / "pointwise_long.csv", index=False)
    pointwise_reasons.to_csv(out_dir / "pointwise_reasons_long.csv", index=False)
    pointwise_summary.to_csv(out_dir / "pointwise_summary.csv", index=False)
    pointwise_respondent_means.to_csv(out_dir / "pointwise_respondent_means.csv", index=False)

    pointwise_latex = make_pointwise_latex(pointwise_summary)
    (out_dir / "human_pointwise_table.tex").write_text(pointwise_latex + "\n", encoding="utf-8")

    pairwise_long = load_pairwise(args.pairwise)
    pairwise_summary, pairwise_reason_summary = summarize_pairwise(pairwise_long)

    pairwise_long.to_csv(out_dir / "pairwise_long.csv", index=False)
    pairwise_summary.to_csv(out_dir / "pairwise_summary.csv", index=False)
    pairwise_reason_summary.to_csv(out_dir / "pairwise_reason_summary.csv", index=False)

    pairwise_audit = (
    pairwise_long
    .groupby(["article_id", "pair_index", "system_a", "system_b"], as_index=False)
    .agg(
        n_votes=("ours_win", "count"),
        ours_votes=("ours_win", "sum"),
        mean_ours_win=("ours_win", "mean"),
        article_a_votes=("choice", lambda s: int((s == 1).sum())),
        article_b_votes=("choice", lambda s: int((s == 2).sum())),
    )
    )

    pairwise_audit["agentic_votes"] = pairwise_audit["n_votes"] - pairwise_audit["ours_votes"]
    pairwise_audit["human_majority"] = np.where(
    pairwise_audit["mean_ours_win"] > 0.5,
    "Ours",
    np.where(pairwise_audit["mean_ours_win"] < 0.5, "Agentic", "Tie"),
    )

    pairwise_audit = pairwise_audit[
        [
        "pair_index",
        "article_id",
        "system_a",
        "system_b",
        "n_votes",
        "article_a_votes",
        "article_b_votes",
        "ours_votes",
        "agentic_votes",
        "mean_ours_win",
        "human_majority",
        ]
    ]

    pairwise_audit.to_csv(out_dir / "pairwise_article_vote_audit.csv", index=False)

    print("\nPairwise article-level vote audit:")
    print(pairwise_audit.to_string(index=False))


    pairwise_latex = make_pairwise_latex(pairwise_summary)
    reason_latex = make_reason_latex(pairwise_reason_summary)
    (out_dir / "human_pairwise_table.tex").write_text(pairwise_latex + "\n", encoding="utf-8")
    (out_dir / "human_pairwise_reason_table.tex").write_text(reason_latex + "\n", encoding="utf-8")

    if args.judge:
        alignment, merged = make_judge_alignment(pointwise_long, args.judge, args.corr, out_dir)
        judge_latex = make_judge_alignment_latex(alignment, args.corr)
        (out_dir / "human_llm_judge_alignment_table.tex").write_text(judge_latex + "\n", encoding="utf-8")
    else:
        alignment = None
    if args.pairwise_judge:
        pairwise_alignment, pairwise_merged = make_pairwise_judge_alignment(
            pairwise_long,
            args.pairwise_judge,
            args.corr,
            out_dir,
        )
        pairwise_judge_latex = make_pairwise_judge_alignment_latex(pairwise_alignment)
        (out_dir / "human_llm_pairwise_alignment_table.tex").write_text(
            pairwise_judge_latex + "\n",
            encoding="utf-8",
        )
    else:
        pairwise_alignment = None

    print("\nHuman pointwise table:")
    print(pointwise_latex)

    print("\nHuman pairwise table:")
    print(pairwise_latex)

    print("\nPairwise reason summary:")
    print(pairwise_reason_summary.to_string(index=False))

    if alignment is not None:
        print("\nHuman--LLM pointwise alignment:")
        print(alignment.to_string(index=False))
    
    if pairwise_alignment is not None:
        print("\nHuman--LLM pairwise alignment:")
        print(pairwise_alignment.to_string(index=False))

    print("\nSaved outputs to:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()

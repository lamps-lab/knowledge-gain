#!/usr/bin/env python3
"""
Build the main KGain correlation table and the human KGain system table.

This version computes:
  1) pooled Spearman/Kendall/Pearson correlations between automatic metrics and
     LLMSim / human normalized KGain;
  2) cluster-bootstrap confidence intervals for pooled rank correlations by
     resampling topics/article_ids;
  3) optional within-abstract rank correlations when each abstract has at least
     --within_min_items candidate systems/articles;
  4) paired Wilcoxon signed-rank tests across topics for Ours vs. Baseline.

Example
-------
python compute_kgain_correlations.py \
  --articles ./eval_dataset_top50.json \
  --reference_col abstract \
  --systems news_0 news_2 \
  --predictions ../src/runs/kgain_run_20260512_140609/predictions.jsonl \
  --judge ./results/open-source/pointwise_top50.json \
  --human_scores ./human/kgain/analysis/participant_article_scores.csv \
  --ours news_0 \
  --baseline news_2 \
  --out_prefix human/kgain/analysis

Notes
-----
- You can pass either participant_article_scores.csv or article_summary.csv to
  --human_scores. The script detects the format automatically.
- With only two systems per topic, within-abstract correlations are usually not
  meaningful; the default --within_min_items 3 prevents reporting degenerate
  within-topic correlations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def read_json_or_jsonl(path: str | Path) -> Any:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".jsonl":
        rows = []
        with path_obj.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path_obj.open("r", encoding="utf-8") as f:
        return json.load(f)


def article_key_from_label(label: Any) -> str:
    label = str(label)
    for key in ["news_0", "news_1", "news_2", "news_3"]:
        if label.startswith(key):
            return key
    return label


def is_correct_label(x: Any) -> int:
    return 1 if str(x).strip().lower() == "correct" else 0


def safe_norm_gain(pre: pd.Series | float, post: pd.Series | float) -> pd.Series | float:
    pre_num = pd.to_numeric(pre, errors="coerce")
    post_num = pd.to_numeric(post, errors="coerce")
    return np.where(
        pre_num < 1.0,
        (post_num - pre_num) / (1.0 - pre_num),
        np.nan,
    )


# -----------------------------------------------------------------------------
# Article metric computation
# -----------------------------------------------------------------------------
def load_articles_long(articles_path: str, systems: list[str], reference_col: str) -> pd.DataFrame:
    data = read_json_or_jsonl(articles_path)
    df = pd.DataFrame(data)

    if "id" not in df.columns:
        raise ValueError(f"Article file must contain an 'id' column. Found: {list(df.columns)}")
    if reference_col not in df.columns:
        raise ValueError(f"Reference column '{reference_col}' not found. Found: {list(df.columns)}")

    rows = []
    for system in systems:
        if system not in df.columns:
            raise ValueError(f"System column '{system}' not found in article file. Found: {list(df.columns)}")

        tmp = df[["id", reference_col, system]].copy()
        tmp = tmp.rename(columns={reference_col: "reference", system: "candidate"})
        tmp["article_id"] = tmp["id"].astype(int)
        tmp["article_key"] = system
        tmp["candidate"] = tmp["candidate"].fillna("").astype(str)
        tmp["reference"] = tmp["reference"].fillna("").astype(str)
        tmp = tmp[(tmp["candidate"].str.strip() != "") & (tmp["reference"].str.strip() != "")]
        rows.append(tmp[["article_id", "article_key", "candidate", "reference"]])

    out = pd.concat(rows, ignore_index=True)
    if out.empty:
        raise RuntimeError("No usable article rows after filtering empty candidates/references.")
    return out


def compute_traditional_metrics(article_long: pd.DataFrame, skip_bertscore: bool = False) -> pd.DataFrame:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rows = []
    for _, row in article_long.iterrows():
        reference = str(row["reference"])
        candidate = str(row["candidate"])
        rouge = scorer.score(reference, candidate)
        bleu = sacrebleu.sentence_bleu(candidate, [reference]).score / 100.0

        rows.append(
            {
                "article_id": int(row["article_id"]),
                "article_key": row["article_key"],
                "rouge1": rouge["rouge1"].fmeasure,
                "rouge2": rouge["rouge2"].fmeasure,
                "rougeL": rouge["rougeL"].fmeasure,
                "bleu": bleu,
            }
        )

    metrics = pd.DataFrame(rows)

    if skip_bertscore:
        metrics["bertscore"] = np.nan
        return metrics

    try:
        from bert_score import score as bertscore_score

        _, _, f1 = bertscore_score(
            article_long["candidate"].tolist(),
            article_long["reference"].tolist(),
            lang="en",
            verbose=True,
        )
        metrics["bertscore"] = [float(x) for x in f1.tolist()]
    except Exception as e:
        print(f"Warning: BERTScore failed; continuing with missing BERTScore. Error: {e}")
        metrics["bertscore"] = np.nan

    return metrics


# -----------------------------------------------------------------------------
# LLMSim and LLM-judge loaders
# -----------------------------------------------------------------------------
def load_llmsim_scores(predictions_path: str, systems: Iterable[str]) -> pd.DataFrame:
    systems = set(systems)
    rows = read_json_or_jsonl(predictions_path)

    records = []
    for rec in rows:
        system = str(rec.get("article_key") or article_key_from_label(rec.get("article_version_label", "")))
        if system not in systems:
            continue

        records.append(
            {
                "article_id": int(rec.get("article_id")),
                "article_key": system,
                "pre_correct": is_correct_label(rec.get("classification_pre")),
                "post_correct": is_correct_label(rec.get("classification_post")),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No LLMSim rows found for systems: {sorted(systems)}")

    out = df.groupby(["article_id", "article_key"], as_index=False).agg(
        llmsim_pre_acc=("pre_correct", "mean"),
        llmsim_post_acc=("post_correct", "mean"),
        llmsim_n=("post_correct", "size"),
    )
    out["llmsim_kgain"] = out["llmsim_post_acc"] - out["llmsim_pre_acc"]
    out["llmsim_norm_kgain"] = safe_norm_gain(out["llmsim_pre_acc"], out["llmsim_post_acc"])
    return out


def load_llm_judge_scores(judge_path: str, systems: Iterable[str]) -> pd.DataFrame:
    systems = list(systems)
    data = read_json_or_jsonl(judge_path)

    rows = []
    for item in data:
        article_id = int(item["id"])
        scores_by_system = item.get("scores_by_system", {})

        for system in systems:
            scores = scores_by_system.get(system)
            if scores is None:
                continue

            rows.append(
                {
                    "article_id": article_id,
                    "article_key": system,
                    "llm_judge_accuracy": scores.get("accuracy"),
                    "llm_judge_completeness": scores.get("completeness"),
                    "llm_judge_relevance": scores.get("relevance"),
                    "llm_judge_clarity": scores.get("clarity"),
                    "llm_judge_expected_kgain": scores.get("knowledge_gain"),
                    "llm_judge_mean_score": scores.get("mean_score"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No LLM judge rows found for systems: {systems}")
    return df


# -----------------------------------------------------------------------------
# Human KGain loader
# -----------------------------------------------------------------------------
def load_human_scores(human_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return article-level human scores and participant-level scores if available.

    Supports either kgain_human_analysis.py outputs:
      - participant_article_scores.csv: pre_acc, post_acc, kgain, normalized_gain
      - article_summary.csv: pre_acc_mean, post_acc_mean, kgain_mean, normalized_gain_mean
    """
    df = pd.read_csv(human_path)

    if "article_id" not in df.columns:
        raise ValueError("Human score file must contain article_id.")

    if "article_key" not in df.columns:
        if "article_version_label" in df.columns:
            df["article_key"] = df["article_version_label"].map(article_key_from_label)
        else:
            raise ValueError("Human score file must contain article_key or article_version_label.")

    # Participant/article format.
    if {"pre_acc", "post_acc", "kgain", "normalized_gain"}.issubset(df.columns):
        participant = df.copy()
        article = participant.groupby(["article_id", "article_key"], as_index=False).agg(
            human_pre_acc=("pre_acc", "mean"),
            human_post_acc=("post_acc", "mean"),
            human_kgain=("kgain", "mean"),
            human_norm_kgain=("normalized_gain", "mean"),
            human_n=("normalized_gain", "size"),
        )
        return article, participant

    # Article summary format from summarize_numeric().
    needed = {"pre_acc_mean", "post_acc_mean", "kgain_mean", "normalized_gain_mean"}
    if needed.issubset(df.columns):
        article = df.rename(
            columns={
                "pre_acc_mean": "human_pre_acc",
                "post_acc_mean": "human_post_acc",
                "kgain_mean": "human_kgain",
                "normalized_gain_mean": "human_norm_kgain",
                "kgain_n": "human_n",
            }
        )
        keep = [
            c
            for c in [
                "article_id",
                "article_key",
                "human_pre_acc",
                "human_post_acc",
                "human_kgain",
                "human_norm_kgain",
                "human_n",
            ]
            if c in article.columns
        ]
        return article[keep].copy(), pd.DataFrame()

    raise ValueError(
        "Unrecognized human score file. Expected participant_article_scores.csv "
        "or article_summary.csv from kgain_human_analysis.py."
    )


# -----------------------------------------------------------------------------
# Correlations
# -----------------------------------------------------------------------------
def metric_specs() -> list[tuple[str, str, str]]:
    return [
        ("ROUGE-1", "rouge1", "Unigram source overlap"),
        ("ROUGE-2", "rouge2", "Bigram source overlap"),
        ("ROUGE-L", "rougeL", "Longest-sequence source overlap"),
        ("BLEU", "bleu", "N-gram precision vs. source"),
        ("BERTScore", "bertscore", "Semantic source similarity"),
        ("LLM judge accuracy", "llm_judge_accuracy", "Factual faithfulness"),
        ("LLM judge completeness", "llm_judge_completeness", "Content coverage"),
        ("LLM judge relevance", "llm_judge_relevance", "Topical usefulness"),
        ("LLM judge clarity", "llm_judge_clarity", "Readability/accessibility"),
        ("LLM judge expected \\textsc{KGain}", "llm_judge_expected_kgain", "Judge-estimated learning"),
    ]


def corr_stats(df: pd.DataFrame, x: str, y: str) -> dict[str, float]:
    sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    out = {"n": len(sub)}

    if len(sub) < 3 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
        return {
            **out,
            "spearman": np.nan,
            "spearman_p": np.nan,
            "kendall": np.nan,
            "kendall_p": np.nan,
            "pearson": np.nan,
            "pearson_p": np.nan,
        }

    if stats is not None:
        sr = stats.spearmanr(sub[x], sub[y])
        kt = stats.kendalltau(sub[x], sub[y])  # tau-b for ties
        pr = stats.pearsonr(sub[x], sub[y])
        return {
            **out,
            "spearman": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
            "kendall": float(kt.statistic),
            "kendall_p": float(kt.pvalue),
            "pearson": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
        }

    return {
        **out,
        "spearman": float(sub[x].corr(sub[y], method="spearman")),
        "spearman_p": np.nan,
        "kendall": float(sub[x].corr(sub[y], method="kendall")),
        "kendall_p": np.nan,
        "pearson": float(sub[x].corr(sub[y], method="pearson")),
        "pearson_p": np.nan,
    }


def corr_value(df: pd.DataFrame, x: str, y: str, method: str) -> float:
    sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()

    if len(sub) < 3 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
        return np.nan

    if stats is not None:
        if method == "spearman":
            return float(stats.spearmanr(sub[x], sub[y]).statistic)
        if method == "kendall":
            return float(stats.kendalltau(sub[x], sub[y]).statistic)
        if method == "pearson":
            return float(stats.pearsonr(sub[x], sub[y]).statistic)

    return float(sub[x].corr(sub[y], method=method))


def bootstrap_corr_ci(
    df: pd.DataFrame,
    x: str,
    y: str,
    method: str,
    group_col: str = "article_id",
    n_boot: int = 10000,
    seed: int = 13,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Cluster-bootstrap a pooled correlation by resampling topics/article_ids.

    This preserves the paired structure of multiple article systems/candidates
    within a topic. It is more appropriate here than resampling individual rows.
    """
    needed = [group_col, x, y]
    if any(c not in df.columns for c in needed):
        return {
            "corr": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_points": 0,
            "n_topics": 0,
            "n_boot_eff": 0,
        }

    work = df[needed].copy()
    work[x] = pd.to_numeric(work[x], errors="coerce")
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work.dropna(subset=[x, y])

    corr = corr_value(work, x, y, method)
    if work.empty:
        return {
            "corr": corr,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_points": 0,
            "n_topics": 0,
            "n_boot_eff": 0,
        }

    groups = {gid: g for gid, g in work.groupby(group_col)}
    group_ids = np.array(list(groups.keys()))
    rng = np.random.default_rng(seed)

    vals = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(group_ids, size=len(group_ids), replace=True)
        sampled = pd.concat([groups[gid] for gid in sampled_ids], ignore_index=True)
        r = corr_value(sampled, x, y, method)
        if pd.notna(r):
            vals.append(r)

    if len(vals) == 0:
        ci_low, ci_high = np.nan, np.nan
    else:
        ci_low, ci_high = np.quantile(vals, [alpha / 2, 1 - alpha / 2])

    return {
        "corr": corr,
        "ci_low": float(ci_low) if pd.notna(ci_low) else np.nan,
        "ci_high": float(ci_high) if pd.notna(ci_high) else np.nan,
        "n_points": int(len(work)),
        "n_topics": int(len(group_ids)),
        "n_boot_eff": int(len(vals)),
    }


def within_abstract_corr_summary(
    df: pd.DataFrame,
    x: str,
    y: str,
    method: str,
    min_items: int = 3,
    n_boot: int = 10000,
    seed: int = 13,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute within-abstract rank correlations and summarize across abstracts.

    This is only meaningful when each abstract has at least min_items candidates.
    With only two systems per abstract, the default min_items=3 prevents
    reporting degenerate within-topic correlations.
    """
    if any(c not in df.columns for c in ["article_id", x, y]):
        return {
            "mean_corr": np.nan,
            "median_corr": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_valid_topics": 0,
        }

    topic_corrs = []
    for article_id, g in df.groupby("article_id"):
        sub = g[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()

        if len(sub) < min_items:
            continue
        if sub[x].nunique() < 2 or sub[y].nunique() < 2:
            continue

        r = corr_value(sub, x, y, method)
        if pd.notna(r):
            topic_corrs.append({"article_id": article_id, "corr": r, "n_items": len(sub)})

    topic_df = pd.DataFrame(topic_corrs)
    if topic_df.empty:
        return {
            "mean_corr": np.nan,
            "median_corr": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_valid_topics": 0,
        }

    corrs = topic_df["corr"].to_numpy(dtype=float)
    mean_corr = float(np.mean(corrs))
    median_corr = float(np.median(corrs))

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(corrs, size=len(corrs), replace=True)
        boot_means.append(float(np.mean(sampled)))

    ci_low, ci_high = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])

    return {
        "mean_corr": mean_corr,
        "median_corr": median_corr,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_valid_topics": int(len(corrs)),
    }


def make_correlation_rows(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for display, col, signal in metric_specs():
        rec = {"Metric": display, "Signal": signal, "metric_col": col}

        for target_name, target_col in [
            ("llmsim", "llmsim_norm_kgain"),
            ("human", "human_norm_kgain"),
        ]:
            stats_dict = corr_stats(merged, col, target_col)
            for k, v in stats_dict.items():
                rec[f"{target_name}_{k}"] = v

        rows.append(rec)

    # Direct LLMSim-human normalized KGain alignment row.
    rec = {
        "Metric": "\\textsc{LLMSim} norm. \\textsc{KGain}",
        "Signal": "Simulated reader learning",
        "metric_col": "llmsim_norm_kgain",
    }

    # LLMSim vs LLMSim is trivial, so leave blank.
    for k in ["n", "spearman", "spearman_p", "kendall", "kendall_p", "pearson", "pearson_p"]:
        rec[f"llmsim_{k}"] = np.nan

    stats_dict = corr_stats(merged, "llmsim_norm_kgain", "human_norm_kgain")
    for k, v in stats_dict.items():
        rec[f"human_{k}"] = v

    rows.append(rec)
    return pd.DataFrame(rows)


def make_bootstrap_and_within_correlation_rows(
    merged: pd.DataFrame,
    n_boot: int,
    seed: int,
    within_min_items: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows = []
    within_rows = []

    targets = [
        ("LLMSim norm. KGain", "llmsim_norm_kgain"),
        ("Human norm. KGain", "human_norm_kgain"),
    ]
    methods = ["spearman", "kendall"]

    for display, col, signal in metric_specs():
        for target_name, target_col in targets:
            for method in methods:
                pooled = bootstrap_corr_ci(
                    merged,
                    col,
                    target_col,
                    method=method,
                    group_col="article_id",
                    n_boot=n_boot,
                    seed=seed,
                )
                pooled_rows.append(
                    {
                        "Metric": display,
                        "Signal": signal,
                        "Target": target_name,
                        "Method": method,
                        **pooled,
                    }
                )

                within = within_abstract_corr_summary(
                    merged,
                    col,
                    target_col,
                    method=method,
                    min_items=within_min_items,
                    n_boot=n_boot,
                    seed=seed,
                )
                within_rows.append(
                    {
                        "Metric": display,
                        "Signal": signal,
                        "Target": target_name,
                        "Method": method,
                        "within_min_items": within_min_items,
                        **within,
                    }
                )

    # Direct LLMSim-human normalized KGain alignment.
    for method in methods:
        pooled = bootstrap_corr_ci(
            merged,
            "llmsim_norm_kgain",
            "human_norm_kgain",
            method=method,
            group_col="article_id",
            n_boot=n_boot,
            seed=seed,
        )
        pooled_rows.append(
            {
                "Metric": "\\textsc{LLMSim} norm. \\textsc{KGain}",
                "Signal": "Simulated reader learning",
                "Target": "Human norm. KGain",
                "Method": method,
                **pooled,
            }
        )

        within = within_abstract_corr_summary(
            merged,
            "llmsim_norm_kgain",
            "human_norm_kgain",
            method=method,
            min_items=within_min_items,
            n_boot=n_boot,
            seed=seed,
        )
        within_rows.append(
            {
                "Metric": "\\textsc{LLMSim} norm. \\textsc{KGain}",
                "Signal": "Simulated reader learning",
                "Target": "Human norm. KGain",
                "Method": method,
                "within_min_items": within_min_items,
                **within,
            }
        )

    return pd.DataFrame(pooled_rows), pd.DataFrame(within_rows)


# -----------------------------------------------------------------------------
# LaTeX helpers
# -----------------------------------------------------------------------------
def fmt(x: float) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{x:.3f}"


def fmt_ci(lo: float, hi: float) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return "[--, --]"
    return f"[{lo:.3f}, {hi:.3f}]"


def latex_corr_table(result_df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{lrrrrl}")
    lines.append(r"\toprule")
    lines.append(
        r"Metric & \multicolumn{2}{c}{\textsc{LLMSim} norm. \textsc{KGain}} "
        r"& \multicolumn{2}{c}{Human norm. \textsc{KGain}} & Signal \\"
    )
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(r" & Spearman $\rho$ & Kendall $\tau_b$ & Spearman $\rho$ & Kendall $\tau_b$ & \\")
    lines.append(r"\midrule")

    for _, row in result_df.iterrows():
        lines.append(
            f"{row['Metric']} & {fmt(row['llmsim_spearman'])} & {fmt(row['llmsim_kendall'])} "
            f"& {fmt(row['human_spearman'])} & {fmt(row['human_kendall'])} & {row['Signal']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Rank correlations between automatic metrics and reader-learning outcomes. "
        r"Traditional metrics are computed against the source abstract. We report Spearman's $\rho$ "
        r"and Kendall's $\tau_b$ because the main question is whether existing metrics preserve the "
        r"ranking of articles by learning. The final row compares simulated and human normalized "
        r"\textsc{KGain}. Pearson correlations and bootstrap confidence intervals are reported in the appendix.}"
    )
    lines.append(r"\label{tab:metrics_vs_kgain}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def latex_pearson_table(result_df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Metric & \textsc{LLMSim} & Human \\")
    lines.append(r"\midrule")

    for _, row in result_df.iterrows():
        lines.append(f"{row['Metric']} & {fmt(row['llmsim_pearson'])} & {fmt(row['human_pearson'])} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Pearson correlations corresponding to Table~\ref{tab:metrics_vs_kgain}.}")
    lines.append(r"\label{tab:metrics_vs_kgain_pearson}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_bootstrap_table(
    pooled_bootstrap_df: pd.DataFrame,
    target: str = "Human norm. KGain",
    method: str = "spearman",
) -> str:
    """Appendix table with bootstrap CIs for one target/method.

    By default this generates the most useful appendix table: Spearman
    correlations with human normalized KGain.
    """
    d = pooled_bootstrap_df[
        (pooled_bootstrap_df["Target"] == target) & (pooled_bootstrap_df["Method"] == method)
    ].copy()

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Corr. & 95\% CI & $n$ \\")
    lines.append(r"\midrule")

    for _, row in d.iterrows():
        lines.append(
            f"{row['Metric']} & {fmt(row['corr'])} & {fmt_ci(row['ci_low'], row['ci_high'])} "
            f"& {int(row['n_points'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    method_label = "Spearman's $\\rho$" if method == "spearman" else "Kendall's $\\tau_b$"
    target_label = target.replace("KGain", r"\textsc{KGain}")

    lines.append(
        rf"\caption{{Cluster-bootstrap confidence intervals for {method_label} correlations with {target_label}. "
        rf"Bootstrap samples resample topics with replacement, preserving the paired article-system structure within each topic.}}"
    )
    lines.append(r"\label{tab:metrics_vs_kgain_bootstrap_ci}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Human system table
# -----------------------------------------------------------------------------
def system_human_table(article_human: pd.DataFrame, ours: str, baseline: str) -> tuple[pd.DataFrame, str]:
    metrics = [
        ("Pre", "human_pre_acc"),
        ("Post", "human_post_acc"),
        ("KGain", "human_kgain"),
        ("NormKGain", "human_norm_kgain"),
    ]

    d = article_human[article_human["article_key"].isin([ours, baseline])].copy()
    pivot = d.pivot_table(
        index="article_id",
        columns="article_key",
        values=[c for _, c in metrics],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{system}" for metric, system in pivot.columns]
    p = pivot.reset_index()

    summary_rows = []
    latex_vals: dict[str, dict[str, str]] = {
        "Baseline": {},
        "Ours": {},
        "$\\Delta$ Ours--Baseline": {},
        "Wilcoxon $p$": {},
    }

    for label, col in metrics:
        c_ours = f"{col}_{ours}"
        c_base = f"{col}_{baseline}"
        sub = p[[c_ours, c_base]].dropna()

        ours_mean = sub[c_ours].mean()
        base_mean = sub[c_base].mean()
        diff = (sub[c_ours] - sub[c_base]).mean()

        if len(sub) >= 2 and stats is not None:
            try:
                wilcox_p = stats.wilcoxon(sub[c_ours], sub[c_base], zero_method="wilcox").pvalue
            except Exception:
                wilcox_p = np.nan
        else:
            wilcox_p = np.nan

        summary_rows.append(
            {
                "metric": label,
                "baseline_mean": base_mean,
                "ours_mean": ours_mean,
                "diff": diff,
                "wilcoxon_p": wilcox_p,
                "n_topics": len(sub),
            }
        )

        latex_vals["Baseline"][label] = fmt(base_mean)
        latex_vals["Ours"][label] = fmt(ours_mean)
        latex_vals["$\\Delta$ Ours--Baseline"][label] = fmt(diff)
        latex_vals["Wilcoxon $p$"][label] = "--" if pd.isna(wilcox_p) else f"{wilcox_p:.3f}".replace("0.", ".")

    def add_row(name: str) -> str:
        return (
            f"{name} & {latex_vals[name]['Pre']} & {latex_vals[name]['Post']} & "
            f"{latex_vals[name]['KGain']} & {latex_vals[name]['NormKGain']} \\\\"
        )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"System & Pre & Post & \textsc{KGain} & Norm. \textsc{KGain} \\")
    lines.append(r"\midrule")
    lines.append(add_row("Baseline"))
    lines.append(add_row("Ours"))
    lines.append(r"\midrule")
    lines.append(add_row("$\\Delta$ Ours--Baseline"))
    lines.append(add_row("Wilcoxon $p$"))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Human pre/post learning evaluation on generated articles. "
        r"Results are topic-level means over paired articles. The \textsc{KGain}-optimized model "
        r"improves post-reading accuracy and normalized \textsc{KGain} over the baseline.}"
    )
    lines.append(r"\label{tab:human_kgain_generated}")
    lines.append(r"\end{table}")

    return pd.DataFrame(summary_rows), "\n".join(lines)


def llmsim_system_test(llmsim: pd.DataFrame, ours: str, baseline: str) -> dict[str, float]:
    """
    Paired topic-level Wilcoxon test for LLMSim normalized KGain.
    Compares Ours vs Baseline across matched article_id topics.
    """
    d = llmsim[llmsim["article_key"].isin([ours, baseline])].copy()

    pivot = d.pivot_table(
        index="article_id",
        columns="article_key",
        values="llmsim_norm_kgain",
        aggfunc="mean",
    ).reset_index()

    sub = pivot[[ours, baseline]].dropna()

    ours_mean = sub[ours].mean()
    baseline_mean = sub[baseline].mean()
    diff = (sub[ours] - sub[baseline]).mean()

    if len(sub) >= 2 and stats is not None:
        try:
            wilcoxon_p = stats.wilcoxon(
                sub[ours],
                sub[baseline],
                zero_method="wilcox",
            ).pvalue
        except Exception:
            wilcoxon_p = np.nan
    else:
        wilcoxon_p = np.nan

    return {
        "baseline_mean": baseline_mean,
        "ours_mean": ours_mean,
        "diff": diff,
        "wilcoxon_p": wilcoxon_p,
        "n_topics": len(sub),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", required=True)
    parser.add_argument("--reference_col", default="abstract")
    parser.add_argument("--systems", nargs="+", default=["news_0", "news_2"])
    parser.add_argument("--predictions", required=True, help="LLMSim predictions JSONL")
    parser.add_argument("--judge", required=True, help="LLM judge JSON")
    parser.add_argument(
        "--human_scores",
        required=True,
        help="participant_article_scores.csv or article_summary.csv from human KGain script",
    )
    parser.add_argument("--ours", default="news_0")
    parser.add_argument("--baseline", default="news_2")
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--out_prefix", default="kgain_metrics")
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=13)
    parser.add_argument(
        "--within_min_items",
        type=int,
        default=3,
        help="Minimum candidates per abstract required to compute within-abstract rank correlations.",
    )

    args = parser.parse_args()

    article_long = load_articles_long(args.articles, args.systems, args.reference_col)
    traditional = compute_traditional_metrics(article_long, skip_bertscore=args.skip_bertscore)
    llmsim = load_llmsim_scores(args.predictions, args.systems)
    judge = load_llm_judge_scores(args.judge, args.systems)
    human_article, human_participant = load_human_scores(args.human_scores)

    merged = traditional.merge(llmsim, on=["article_id", "article_key"], how="inner")
    merged = merged.merge(judge, on=["article_id", "article_key"], how="left")
    merged = merged.merge(human_article, on=["article_id", "article_key"], how="inner")

    if merged.empty:
        raise RuntimeError("Merged dataframe is empty. Check article IDs/system keys across inputs.")

    result_df = make_correlation_rows(merged)
    pooled_bootstrap_df, within_abstract_df = make_bootstrap_and_within_correlation_rows(
        merged,
        n_boot=args.bootstrap_iters,
        seed=args.bootstrap_seed,
        within_min_items=args.within_min_items,
    )
    system_summary, human_table_tex = system_human_table(human_article, args.ours, args.baseline)

    prefix = Path(args.out_prefix)

    merged.to_csv(f"{prefix}_merged_metric_points.csv", index=False)
    result_df.to_csv(f"{prefix}_rank_correlations_full.csv", index=False)
    pooled_bootstrap_df.to_csv(f"{prefix}_rank_correlations_bootstrap_ci.csv", index=False)
    within_abstract_df.to_csv(f"{prefix}_within_abstract_rank_correlations.csv", index=False)
    system_summary.to_csv(f"{prefix}_human_kgain_system_table.csv", index=False)

    llmsim_summary = llmsim_system_test(llmsim, args.ours, args.baseline)
    pd.DataFrame([llmsim_summary]).to_csv(
        f"{prefix}_llmsim_kgain_system_test.csv",
        index=False,
    )
    prefix = Path(args.out_prefix)
    merged.to_csv(f"{prefix}_merged_metric_points.csv", index=False)
    result_df.to_csv(f"{prefix}_rank_correlations_full.csv", index=False)
    system_summary.to_csv(f"{prefix}_human_kgain_system_table.csv", index=False)
    
    print("\nLLMSim system test:")
    print(pd.DataFrame([llmsim_summary]).to_string(index=False))


    with open(f"{prefix}_metrics_vs_kgain_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_corr_table(result_df) + "\n")

    with open(f"{prefix}_metrics_vs_kgain_pearson_appendix.tex", "w", encoding="utf-8") as f:
        f.write(latex_pearson_table(result_df) + "\n")

    with open(f"{prefix}_metrics_vs_kgain_bootstrap_ci_appendix.tex", "w", encoding="utf-8") as f:
        f.write(latex_bootstrap_table(pooled_bootstrap_df, target="Human norm. KGain", method="spearman") + "\n")

    with open(f"{prefix}_human_kgain_generated_table.tex", "w", encoding="utf-8") as f:
        f.write(human_table_tex + "\n")

    print("\nMerged rows:", len(merged))
    print("Systems:", ", ".join(args.systems))

    print("\nHuman system table:")
    print(system_summary.to_string(index=False))

    print("\nCorrelation table:")
    display_cols = ["Metric", "llmsim_spearman", "llmsim_kendall", "human_spearman", "human_kendall", "Signal"]
    print(result_df[display_cols].to_string(index=False))

    print("\nBootstrap CI table preview: Human norm. KGain, Spearman")
    preview = pooled_bootstrap_df[
        (pooled_bootstrap_df["Target"] == "Human norm. KGain")
        & (pooled_bootstrap_df["Method"] == "spearman")
    ][["Metric", "corr", "ci_low", "ci_high", "n_points", "n_topics", "n_boot_eff"]]
    print(preview.to_string(index=False))

    print("\nWithin-abstract correlation preview:")
    print(within_abstract_df.head(20).to_string(index=False))

    print("\nWrote:")
    print(f"  {prefix}_merged_metric_points.csv")
    print(f"  {prefix}_rank_correlations_full.csv")
    print(f"  {prefix}_rank_correlations_bootstrap_ci.csv")
    print(f"  {prefix}_within_abstract_rank_correlations.csv")
    print(f"  {prefix}_human_kgain_system_table.csv")
    print(f"  {prefix}_metrics_vs_kgain_table.tex")
    print(f"  {prefix}_metrics_vs_kgain_pearson_appendix.tex")
    print(f"  {prefix}_metrics_vs_kgain_bootstrap_ci_appendix.tex")
    print(f"  {prefix}_human_kgain_generated_table.tex")


if __name__ == "__main__":
    main()

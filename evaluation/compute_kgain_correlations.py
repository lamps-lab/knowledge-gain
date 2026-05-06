#!/usr/bin/env python3
"""
Build the main KGain correlation table and the human KGain system table.

python compute_kgain_correlations.py   --articles ./eval_dataset_top50.json   --reference_col abstract   --systems news_0 news_2   --predictions ../src/runs/kgain_run_20260512_140609/predictions.jsonl   --judge ./results/open-source/pointwise_top50.json   --human_scores ./human/kgain/analysis/participant_article_scores.csv   --ours news_0   --baseline news_2   --out_prefix human/kgain/analysis

You can pass either participant_article_scores.csv or article_summary.csv to
--human_scores. The script detects the format automatically.
"""

from __future__ import annotations

import argparse
import json
import math
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
    return np.where(pd.to_numeric(pre, errors="coerce") < 1.0,
                    (pd.to_numeric(post, errors="coerce") - pd.to_numeric(pre, errors="coerce")) /
                    (1.0 - pd.to_numeric(pre, errors="coerce")),
                    np.nan)

# Article metrics
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
        rows.append({
            "article_id": int(row["article_id"]),
            "article_key": row["article_key"],
            "rouge1": rouge["rouge1"].fmeasure,
            "rouge2": rouge["rouge2"].fmeasure,
            "rougeL": rouge["rougeL"].fmeasure,
            "bleu": bleu,
        })
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

# LLMSim and LLM-judge loaders
def load_llmsim_scores(predictions_path: str, systems: Iterable[str]) -> pd.DataFrame:
    systems = set(systems)
    rows = read_json_or_jsonl(predictions_path)
    records = []
    for rec in rows:
        system = str(rec.get("article_key") or article_key_from_label(rec.get("article_version_label", "")))
        if system not in systems:
            continue
        records.append({
            "article_id": int(rec.get("article_id")),
            "article_key": system,
            "pre_correct": is_correct_label(rec.get("classification_pre")),
            "post_correct": is_correct_label(rec.get("classification_post")),
        })
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
            rows.append({
                "article_id": article_id,
                "article_key": system,
                "llm_judge_accuracy": scores.get("accuracy"),
                "llm_judge_completeness": scores.get("completeness"),
                "llm_judge_relevance": scores.get("relevance"),
                "llm_judge_clarity": scores.get("clarity"),
                "llm_judge_expected_kgain": scores.get("knowledge_gain"),
                "llm_judge_mean_score": scores.get("mean_score"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No LLM judge rows found for systems: {systems}")
    return df

# Human KGain loader
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
        article = df.rename(columns={
            "pre_acc_mean": "human_pre_acc",
            "post_acc_mean": "human_post_acc",
            "kgain_mean": "human_kgain",
            "normalized_gain_mean": "human_norm_kgain",
            "kgain_n": "human_n",
        })
        keep = [c for c in ["article_id", "article_key", "human_pre_acc", "human_post_acc", "human_kgain", "human_norm_kgain", "human_n"] if c in article.columns]
        return article[keep].copy(), pd.DataFrame()

    raise ValueError(
        "Unrecognized human score file. Expected participant_article_scores.csv "
        "or article_summary.csv from kgain_human_analysis.py."
    )

# Correlations and LaTeX
def corr_stats(df: pd.DataFrame, x: str, y: str) -> dict[str, float]:
    sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    out = {"n": len(sub)}
    if len(sub) < 3 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
        return {**out, "spearman": np.nan, "kendall": np.nan, "pearson": np.nan,
                "spearman_p": np.nan, "kendall_p": np.nan, "pearson_p": np.nan}
    if stats is not None:
        sr = stats.spearmanr(sub[x], sub[y])
        kt = stats.kendalltau(sub[x], sub[y])  # tau-b for ties
        pr = stats.pearsonr(sub[x], sub[y])
        return {
            **out,
            "spearman": float(sr.statistic), "spearman_p": float(sr.pvalue),
            "kendall": float(kt.statistic), "kendall_p": float(kt.pvalue),
            "pearson": float(pr.statistic), "pearson_p": float(pr.pvalue),
        }
    return {
        **out,
        "spearman": float(sub[x].corr(sub[y], method="spearman")), "spearman_p": np.nan,
        "kendall": float(sub[x].corr(sub[y], method="kendall")), "kendall_p": np.nan,
        "pearson": float(sub[x].corr(sub[y], method="pearson")), "pearson_p": np.nan,
    }


def fmt(x: float) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{x:.3f}"


def make_correlation_rows(merged: pd.DataFrame) -> pd.DataFrame:
    metric_specs = [
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

    rows = []
    for display, col, signal in metric_specs:
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
    rec = {"Metric": "\\textsc{LLMSim} norm. \\textsc{KGain}",
           "Signal": "Simulated reader learning",
           "metric_col": "llmsim_norm_kgain"}
    # LLMSim vs LLMSim is trivial, so leave blank.
    for k in ["n", "spearman", "spearman_p", "kendall", "kendall_p", "pearson", "pearson_p"]:
        rec[f"llmsim_{k}"] = np.nan
    stats_dict = corr_stats(merged, "llmsim_norm_kgain", "human_norm_kgain")
    for k, v in stats_dict.items():
        rec[f"human_{k}"] = v
    rows.append(rec)

    return pd.DataFrame(rows)


def latex_corr_table(result_df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{lrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Metric & \multicolumn{2}{c}{\textsc{LLMSim} norm. \textsc{KGain}} & \multicolumn{2}{c}{Human norm. \textsc{KGain}} & Signal \\")
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
        r"\textsc{KGain}. Pearson correlations are reported in the appendix.}"
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

# Human system table
def system_human_table(article_human: pd.DataFrame, ours: str, baseline: str) -> tuple[pd.DataFrame, str]:
    metrics = [
        ("Pre", "human_pre_acc"),
        ("Post", "human_post_acc"),
        ("KGain", "human_kgain"),
        ("NormKGain", "human_norm_kgain"),
    ]
    d = article_human[article_human["article_key"].isin([ours, baseline])].copy()
    pivot = d.pivot_table(index="article_id", columns="article_key", values=[c for _, c in metrics], aggfunc="mean")
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
        summary_rows.append({
            "metric": label,
            "baseline_mean": base_mean,
            "ours_mean": ours_mean,
            "diff": diff,
            "wilcoxon_p": wilcox_p,
            "n_topics": len(sub),
        })
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


# Main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", required=True)
    parser.add_argument("--reference_col", default="abstract")
    parser.add_argument("--systems", nargs="+", default=["news_0", "news_2"])
    parser.add_argument("--predictions", required=True, help="LLMSim predictions JSONL")
    parser.add_argument("--judge", required=True, help="LLM judge JSON")
    parser.add_argument("--human_scores", required=True, help="participant_article_scores.csv or article_summary.csv from human KGain script")
    parser.add_argument("--ours", default="news_0")
    parser.add_argument("--baseline", default="news_2")
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--out_prefix", default="kgain_metrics")
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
    system_summary, human_table_tex = system_human_table(human_article, args.ours, args.baseline)

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
    with open(f"{prefix}_human_kgain_generated_table.tex", "w", encoding="utf-8") as f:
        f.write(human_table_tex + "\n")

    print("\nMerged rows:", len(merged))
    print("Systems:", ", ".join(args.systems))
    print("\nHuman system table:")
    print(system_summary.to_string(index=False))
    print("\nCorrelation table:")
    display_cols = ["Metric", "llmsim_spearman", "llmsim_kendall", "human_spearman", "human_kendall", "Signal"]
    print(result_df[display_cols].to_string(index=False))
    print("\nWrote:")
    print(f"  {prefix}_merged_metric_points.csv")
    print(f"  {prefix}_rank_correlations_full.csv")
    print(f"  {prefix}_human_kgain_system_table.csv")
    print(f"  {prefix}_metrics_vs_kgain_table.tex")
    print(f"  {prefix}_metrics_vs_kgain_pearson_appendix.tex")
    print(f"  {prefix}_human_kgain_generated_table.tex")


if __name__ == "__main__":
    main()

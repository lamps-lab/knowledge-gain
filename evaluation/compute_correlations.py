#!/usr/bin/env python3
"""
python3 compute_correlations.py   --articles ./eval_dataset_top50.json   --candidate_col news_0   --reference_col abstract   --predictions ../src/runs/kgain_run_20260512_140609/predictions.jsonl   --judge results/open-source/pointwise_top50.json   --corr spearman   --out_prefix news0_vs_abstract_spearman
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer


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


def is_correct_label(x: Any) -> int:
    return 1 if str(x).strip().lower() == "correct" else 0


def system_from_version_label(label: str) -> str:
    label = str(label)

    if label.startswith("news_0"):
        return "news_0"
    if label.startswith("news_1"):
        return "news_1"
    if label.startswith("news_2"):
        return "news_2"
    if label.startswith("news_3"):
        return "news_3"

    return label


def load_articles(
    articles_path: str,
    candidate_col: str,
    reference_col: str,
) -> pd.DataFrame:
    data = read_json_or_jsonl(articles_path)
    df = pd.DataFrame(data)

    if "id" not in df.columns:
        raise ValueError(
            f"Article file must contain an 'id' column. Found columns: {list(df.columns)}"
        )

    if candidate_col not in df.columns:
        raise ValueError(
            f"Candidate column '{candidate_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if reference_col not in df.columns:
        raise ValueError(
            f"Reference column '{reference_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[["id", candidate_col, reference_col]].copy()
    out = out.rename(
        columns={
            candidate_col: "candidate",
            reference_col: "reference",
        }
    )

    out["id"] = out["id"].astype(int)
    out["candidate"] = out["candidate"].fillna("").astype(str)
    out["reference"] = out["reference"].fillna("").astype(str)

    out = out[
        (out["candidate"].str.strip().str.len() > 0)
        & (out["reference"].str.strip().str.len() > 0)
    ].copy()

    if out.empty:
        raise RuntimeError("No usable article rows after filtering empty candidate/reference texts.")

    return out


def compute_traditional_metrics(
    article_df: pd.DataFrame,
    skip_bertscore: bool = False,
) -> pd.DataFrame:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    rows = []

    for _, row in article_df.iterrows():
        article_id = int(row["id"])
        candidate = str(row["candidate"])
        reference = str(row["reference"])

        # ROUGE expects: target/reference first, prediction/candidate second.
        rouge = scorer.score(reference, candidate)

        # sacreBLEU expects: hypothesis/candidate first, references second.
        bleu = sacrebleu.sentence_bleu(
            candidate,
            [reference],
        ).score / 100.0

        rows.append(
            {
                "id": article_id,
                "rouge1": rouge["rouge1"].fmeasure,
                "rouge2": rouge["rouge2"].fmeasure,
                "rougeL": rouge["rougeL"].fmeasure,
                "bleu": bleu,
            }
        )

    metrics_df = pd.DataFrame(rows)

    if skip_bertscore:
        metrics_df["bertscore"] = math.nan
        return metrics_df

    try:
        from bert_score import score as bertscore_score

        candidates = article_df["candidate"].tolist()
        references = article_df["reference"].tolist()

        _, _, f1 = bertscore_score(
            candidates,
            references,
            lang="en",
            verbose=True,
        )

        metrics_df["bertscore"] = [float(x) for x in f1.tolist()]

    except Exception as e:
        print(f"Warning: BERTScore failed. Continuing with BERTScore as missing. Error: {e}")
        metrics_df["bertscore"] = math.nan

    return metrics_df


def load_llmsim_news0_kgain(predictions_path: str) -> pd.DataFrame:
    rows = read_json_or_jsonl(predictions_path)

    records = []

    for rec in rows:
        if "article_key" in rec:
            system = str(rec["article_key"])
        else:
            system = system_from_version_label(rec["article_version_label"])

        if system != "news_0":
            continue

        records.append(
            {
                "id": int(rec["article_id"]),
                "pre_correct": is_correct_label(rec["classification_pre"]),
                "post_correct": is_correct_label(rec["classification_post"]),
            }
        )

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No news_0 rows found in LLMSim predictions.")

    out = (
        df.groupby("id", as_index=False)
        .agg(
            llmsim_pre_correct=("pre_correct", "mean"),
            llmsim_post_correct=("post_correct", "mean"),
            llmsim_n=("post_correct", "size"),
        )
    )

    out["llmsim_kgain"] = out["llmsim_post_correct"] - out["llmsim_pre_correct"]

    return out


def load_llm_judge_news0(judge_path: str) -> pd.DataFrame:
    data = read_json_or_jsonl(judge_path)

    rows = []

    for item in data:
        article_id = int(item["id"])
        scores = item.get("scores_by_system", {}).get("news_0")

        if scores is None:
            continue

        rows.append(
            {
                "id": article_id,
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
        raise RuntimeError("No news_0 rows found in LLM judge file.")

    return df


def corr_pair(df: pd.DataFrame, x: str, y: str, method: str) -> float:
    if x not in df.columns or y not in df.columns:
        return math.nan

    sub = df[[x, y]].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna()

    if len(sub) < 2:
        return math.nan

    if sub[x].nunique() < 2 or sub[y].nunique() < 2:
        return math.nan

    return float(sub[x].corr(sub[y], method=method))


def fmt_corr(x: float) -> str:
    if x is None or math.isnan(x):
        return "--"
    return f"{x:.3f}"


def make_latex_table(result_df: pd.DataFrame, corr_name: str) -> str:
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{lrl}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Corr. with \textsc{LLMSim} KGain & Signal \\")
    lines.append(r"\midrule")

    for _, row in result_df.iterrows():
        metric = row["Metric"]
        corr = row["Corr. with LLMSim KGain"]
        signal = row["Signal"]
        lines.append(f"{metric} & {corr} & {signal} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{Correlation between automatic metrics and article-level "
        rf"\textsc{{LLMSim}} \textsc{{KnowledgeGain}} for generated \texttt{{news\_0}} articles. "
        rf"Traditional metrics are computed against the source abstract. "
        rf"We report {corr_name} correlation.}}"
    )
    lines.append(r"\label{tab:metrics_vs_llmsim_kgain}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--articles",
        required=True,
        help="JSON/JSONL file containing article texts, e.g. ../evaluation/eval_dataset.json",
    )

    parser.add_argument(
        "--candidate_col",
        default="news_0",
        help="Generated article column. Default: news_0.",
    )

    parser.add_argument(
        "--reference_col",
        default="abstract",
        help="Source abstract column. Default: abstract.",
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to LLMSim predictions.jsonl.",
    )

    parser.add_argument(
        "--judge",
        required=True,
        help="Path to LLM judge JSON file.",
    )

    parser.add_argument(
        "--corr",
        choices=["pearson", "spearman", "kendall"],
        default="pearson",
        help="Correlation type. Default: pearson.",
    )

    parser.add_argument(
        "--skip_bertscore",
        action="store_true",
        help="Skip BERTScore if dependencies/model download are inconvenient.",
    )

    parser.add_argument(
        "--out_prefix",
        default="news0_vs_abstract",
    )

    args = parser.parse_args()

    article_df = load_articles(
        articles_path=args.articles,
        candidate_col=args.candidate_col,
        reference_col=args.reference_col,
    )

    traditional_df = compute_traditional_metrics(
        article_df=article_df,
        skip_bertscore=args.skip_bertscore,
    )

    llmsim_df = load_llmsim_news0_kgain(args.predictions)
    judge_df = load_llm_judge_news0(args.judge)

    merged = (
        llmsim_df
        .merge(traditional_df, on="id", how="inner")
        .merge(judge_df, on="id", how="left")
    )

    if merged.empty:
        raise RuntimeError("Merged dataframe is empty. Check article IDs across files.")

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
        ("LLM judge expected KGain", "llm_judge_expected_kgain", "Judge-estimated learning"),
    ]

    result_rows = []

    for display_name, col, signal in metric_specs:
        corr = corr_pair(merged, col, "llmsim_kgain", args.corr)

        result_rows.append(
            {
                "Metric": display_name,
                "Corr. with LLMSim KGain": fmt_corr(corr),
                "Signal": signal,
                "raw_corr": corr,
            }
        )

    result_df = pd.DataFrame(result_rows)

    metrics_path = f"{args.out_prefix}_traditional_metrics.csv"
    merged_path = f"{args.out_prefix}_merged_points.csv"
    table_csv_path = f"{args.out_prefix}_metric_correlations.csv"
    table_tex_path = f"{args.out_prefix}_metric_correlations.tex"

    traditional_df.to_csv(metrics_path, index=False)
    merged.to_csv(merged_path, index=False)
    result_df[["Metric", "Corr. with LLMSim KGain", "Signal"]].to_csv(
        table_csv_path,
        index=False,
    )

    latex = make_latex_table(
        result_df[["Metric", "Corr. with LLMSim KGain", "Signal"]],
        corr_name=args.corr,
    )

    with open(table_tex_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")

    print("\nData points:")
    print(f"  candidate column: {args.candidate_col}")
    print(f"  reference column: {args.reference_col}")
    print(f"  articles with all required data: {len(merged)}")
    print(f"  mean LLMSim pre correct:  {merged['llmsim_pre_correct'].mean():.3f}")
    print(f"  mean LLMSim post correct: {merged['llmsim_post_correct'].mean():.3f}")
    print(f"  mean LLMSim KGain:        {merged['llmsim_kgain'].mean():.3f}")

    print("\nCorrelation table:")
    print(result_df[["Metric", "Corr. with LLMSim KGain", "Signal"]].to_string(index=False))

    print("\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {merged_path}")
    print(f"  {table_csv_path}")
    print(f"  {table_tex_path}")


if __name__ == "__main__":
    main()
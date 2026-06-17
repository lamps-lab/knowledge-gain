#!/usr/bin/env python3

import json
import os
from typing import Any, Dict, List

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "llm_judge_pairwise_news0_vs_news2_claude_sonnet.json"
OUTDIR = "pairwise_plots"

NEWS0 = "news_0"
NEWS1 = "news_2"

os.makedirs(OUTDIR, exist_ok=True)


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def winner_to_result_from_news0(winner_key):
    if winner_key == NEWS0:
        return "win"
    if winner_key == NEWS1:
        return "loss"
    return "tie"


def result_to_score(result: str) -> float:
    if result == "win":
        return 1.0
    if result == "loss":
        return 0.0
    return 0.5


def main():
    data = load_json(INPUT_FILE)

    rows = []
    skipped = 0

    for rec in data:
        if rec.get("comparison") is None and rec.get("winner_key") is None and rec.get("judgment") is None:
            skipped += 1
            continue

        winner_key = rec.get("winner_key")
        result_news0 = winner_to_result_from_news0(winner_key)
        score_news0 = result_to_score(result_news0)
        score_news1 = 1.0 - score_news0 if result_news0 != "tie" else 0.5

        rows.append(
            {
                "id": rec.get("id"),
                "date": rec.get("date"),
                "category": rec.get("category"),
                "winner_key": winner_key,
                "winner_label": rec.get("winner_label"),
                "result_news0": result_news0,
                "score_news0": score_news0,
                "score_news1": score_news1,
                "reason": (rec.get("judgment") or {}).get("reason"),
                "presented_article_a": (rec.get("presented_order") or {}).get("article_a"),
                "presented_article_b": (rec.get("presented_order") or {}).get("article_b"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "pairwise_news0_vs_news1_flat.csv"), index=False)

    if df.empty:
        print("No valid comparisons found.")
        return

    # Parse dates if possible
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")

    # --------------------------------------------------
    # 1) Overall counts
    # --------------------------------------------------
    overall_counts = (
        df["result_news0"]
        .value_counts()
        .reindex(["win", "loss", "tie"], fill_value=0)
    )

    plt.figure(figsize=(6, 4))
    overall_counts.plot(kind="bar")
    plt.title("Overall pairwise results (news_0 vs news_1)")
    plt.xlabel("Result from news_0 perspective")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, "overall_result_counts.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # 2) Overall scores by system
    # --------------------------------------------------
    system_summary = pd.DataFrame(
        [
            {"system": NEWS0, "mean_score": df["score_news0"].mean(), "total_score": df["score_news0"].sum()},
            {"system": NEWS1, "mean_score": df["score_news1"].mean(), "total_score": df["score_news1"].sum()},
        ]
    )

    plt.figure(figsize=(5, 4))
    plt.bar(system_summary["system"], system_summary["mean_score"])
    plt.title("Mean pairwise score by system")
    plt.xlabel("System")
    plt.ylabel("Mean score (win=1, tie=0.5, loss=0)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, "mean_score_by_system.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # 3) Mean score by category
    # --------------------------------------------------
    score_by_category = (
        df.groupby("category", dropna=False)["score_news0"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 4))
    score_by_category.plot(kind="bar")
    plt.title("Mean pairwise score of news_0 by category")
    plt.xlabel("Category")
    plt.ylabel("Mean score (win=1, tie=0.5, loss=0)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, "mean_score_news0_by_category.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # 4) Strict win rate by category (ties excluded)
    # --------------------------------------------------
    df_notie = df[df["result_news0"] != "tie"].copy()
    if not df_notie.empty:
        df_notie["win_news0"] = (df_notie["result_news0"] == "win").astype(int)

        win_rate_by_category = (
            df_notie.groupby("category", dropna=False)["win_news0"]
            .mean()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(8, 4))
        win_rate_by_category.plot(kind="bar")
        plt.title("Strict win rate of news_0 by category (ties excluded)")
        plt.xlabel("Category")
        plt.ylabel("Win rate")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTDIR, "strict_win_rate_news0_by_category.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    # --------------------------------------------------
    # 5) Stacked category counts
    # --------------------------------------------------
    cat_counts = pd.crosstab(df["category"], df["result_news0"]).reindex(
        columns=["win", "loss", "tie"], fill_value=0
    )

    plt.figure(figsize=(9, 5))
    cat_counts.plot(kind="bar", stacked=True)
    plt.title("Pairwise results by category (news_0 perspective)")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, "stacked_results_by_category.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # 6) Over time, if dates are present
    # --------------------------------------------------
    df_dates = df.dropna(subset=["date_parsed"]).copy()
    if not df_dates.empty and df_dates["date_parsed"].nunique() > 1:
        score_by_date = (
            df_dates.groupby("date_parsed")["score_news0"]
            .mean()
            .sort_index()
        )

        plt.figure(figsize=(8, 4))
        plt.plot(score_by_date.index, score_by_date.values, marker="o")
        plt.title("Mean pairwise score of news_0 over time")
        plt.xlabel("Date")
        plt.ylabel("Mean score (win=1, tie=0.5, loss=0)")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTDIR, "mean_score_news0_over_time.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    # --------------------------------------------------
    # 7) Summary CSV
    # --------------------------------------------------
    n_total = len(df)
    n_win = int((df["result_news0"] == "win").sum())
    n_loss = int((df["result_news0"] == "loss").sum())
    n_tie = int((df["result_news0"] == "tie").sum())

    summary = pd.DataFrame(
        [
            {
                "n_total": n_total,
                "n_news0_wins": n_win,
                "n_news1_wins": n_loss,
                "n_ties": n_tie,
                "news0_mean_score": df["score_news0"].mean(),
                "news1_mean_score": df["score_news1"].mean(),
                "skipped_records": skipped,
            }
        ]
    )
    summary.to_csv(os.path.join(OUTDIR, "summary_metrics.csv"), index=False)

    # System summary CSV too
    system_summary.to_csv(os.path.join(OUTDIR, "summary_by_system.csv"), index=False)

    print(f"Saved outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Human KnowledgeGain analysis for G3a/G3b Qualtrics exports.

What it does
------------
1) Reads G3a/G3b Qualtrics CSVs, skipping the two Qualtrics metadata rows.
2) Uses the supplied answer-key JSON to score 6 pre and 6 post questions per article.
3) Produces row-level human annotations compatible with LLMSim-style JSONL fields.
4) Aggregates pre, post, and KnowledgeGain (post - pre) by participant/article,
   article/version, and system (news_0 vs news_2).
5) Produces plots for pre/post/KGain distributions and article-level paired KGain.
6) Optionally merges an LLMSim JSONL file and computes correlations.

Expected Qualtrics layout
-------------------------
For each article order k=1..20, Qualtrics uses 12 question columns:
    pre:  Q(12*(k-1)+1) ... Q(12*(k-1)+6)
    post: Q(12*(k-1)+7) ... Q(12*(k-1)+12)
The reading timer columns are between pre and post blocks, but are not needed for scoring.

Example
-------
python kgain_human_analysis.py \
  --g3a g3a.csv --g3b g3b.csv --answer_key_json answer_key.json \
  --outdir kgain_outputs

python kgain_human_analysis.py \
  --g3a g3a.csv --g3b g3b.csv --answer_key_json answer_key.json \
  --llmsim_jsonl llmsim_predictions.jsonl \
  --outdir kgain_outputs
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


# -----------------------------------------------------------------------------
# Study design supplied by the user
# -----------------------------------------------------------------------------
ORDER_MAP = {
    "G3a": [
        (1, 29733, "B", "Health", "news_2b"),
        (2, 29734, "B", "Space", "news_2b"),
        (3, 29739, "A", "Health", "news_0a"),
        (4, 29753, "B", "Health", "news_2b"),
        (5, 29775, "A", "Tech", "news_0a"),
        (6, 29802, "A", "Humans", "news_0a"),
        (7, 29846, "B", "Nature", "news_2b"),
        (8, 29866, "A", "Nature", "news_0a"),
        (9, 29873, "A", "Humans", "news_0a"),
        (10, 29902, "A", "Nature", "news_0a"),
        (11, 29929, "A", "Physics", "news_0a"),
        (12, 29943, "B", "Tech", "news_2b"),
        (13, 29958, "B", "Physics", "news_2b"),
        (14, 29988, "B", "Humans", "news_2b"),
        (15, 30052, "A", "Physics", "news_0a"),
        (16, 30061, "B", "Physics", "news_2b"),
        (17, 30086, "A", "Space", "news_0a"),
        (18, 30184, "A", "Tech", "news_0a"),
        (19, 30214, "B", "Space", "news_2b"),
        (20, 30708, "B", "Tech", "news_2b"),
    ],
    "G3b": [
        (1, 29733, "B", "Health", "news_0b"),
        (2, 29734, "B", "Space", "news_0b"),
        (3, 29739, "A", "Health", "news_2a"),
        (4, 29753, "B", "Health", "news_0b"),
        (5, 29775, "A", "Tech", "news_2a"),
        (6, 29802, "A", "Humans", "news_2a"),
        (7, 29846, "B", "Nature", "news_0b"),
        (8, 29866, "A", "Nature", "news_2a"),
        (9, 29873, "A", "Humans", "news_2a"),
        (10, 29902, "A", "Nature", "news_2a"),
        (11, 29929, "A", "Physics", "news_2a"),
        (12, 29943, "B", "Tech", "news_0b"),
        (13, 29958, "B", "Physics", "news_0b"),
        (14, 29988, "B", "Humans", "news_0b"),
        (15, 30052, "A", "Physics", "news_2a"),
        (16, 30061, "B", "Physics", "news_0b"),
        (17, 30086, "A", "Space", "news_2a"),
        (18, 30184, "A", "Tech", "news_2a"),
        (19, 30214, "B", "Space", "news_0b"),
        (20, 30708, "B", "Tech", "news_0b"),
    ],
}

EXPECTED_N = {"G3a": 16, "G3b": 16}


def article_key_from_label(version_label: str) -> str:
    if version_label.startswith("news_0"):
        return "news_0"
    if version_label.startswith("news_2"):
        return "news_2"
    raise ValueError(f"Unknown article_version_label: {version_label}")


def clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x)).strip()


def parse_answer(x: Any) -> float:
    """Return numeric option code as float/NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    # Qualtrics values are usually numeric strings like "1".
    try:
        return int(float(s))
    except ValueError:
        # Fallback for accidental labels such as "True" or "I do not know...".
        return np.nan


def classify_answer(ans: Any, correct_option: int, idk_option: int) -> str:
    val = parse_answer(ans)
    if pd.isna(val):
        return "missing"
    val = int(val)
    if val == correct_option:
        return "correct"
    if val == idk_option:
        return "dk"
    return "incorrect"


def correct_flag(classification: str) -> float:
    if classification == "missing":
        return np.nan
    return 1.0 if classification == "correct" else 0.0


def get_question_columns_in_display_order(df: pd.DataFrame) -> List[str]:
    """Return the 240 Qualtrics question columns in display order.

    Important: one uploaded Qualtrics export has a duplicated display name
    (Q141 appears twice and Q143 is absent), so name-based Q-number lookup is
    unsafe. Pandas preserves duplicate columns by appending .1, and the column
    order is still correct. We therefore map logical Q1..Q240 to the 1st..240th
    question-like column in the file.
    """
    qcols = [c for c in df.columns if re.match(r"^Q\d+\.", str(c))]
    if len(qcols) != 240:
        raise ValueError(f"Expected 240 question columns, found {len(qcols)}")
    return qcols


def get_q_column(qcols: Sequence[str], qnum: int) -> str:
    if qnum < 1 or qnum > len(qcols):
        raise KeyError(f"Logical Q{qnum} is outside the available question columns")
    return qcols[qnum - 1]


def qnum_for(order: int, question_in_set: int, phase: str) -> int:
    if phase not in {"pre", "post"}:
        raise ValueError("phase must be pre or post")
    offset = 0 if phase == "pre" else 6
    return 12 * (order - 1) + offset + question_in_set


def mean_ci95(series: pd.Series) -> Tuple[float, float, float, float, int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = int(s.shape[0])
    mean = float(s.mean()) if n else np.nan
    sd = float(s.std(ddof=1)) if n > 1 else np.nan
    se = float(sd / math.sqrt(n)) if n > 1 else np.nan
    ci = float(1.96 * se) if n > 1 else np.nan
    return mean, sd, se, ci, n


def summarize_numeric(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    records = []
    grouped = df.groupby(group_cols, dropna=False)
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        rec = {col: val for col, val in zip(group_cols, key)}
        for col in value_cols:
            mean, sd, se, ci, n = mean_ci95(g[col])
            rec[f"{col}_mean"] = mean
            rec[f"{col}_sd"] = sd
            rec[f"{col}_se"] = se
            rec[f"{col}_ci95"] = ci
            rec[f"{col}_n"] = n
        records.append(rec)
    return pd.DataFrame(records)


def write_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Write CSVs robustly even when text fields contain quotes/newlines."""
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        doublequote=True,
        lineterminator="\n",
    )


def _coerce_answer_key_records(raw: Any) -> List[Dict[str, Any]]:
    """Accept common answer-key JSON shapes and return article records.

    Supported shapes:
      1. [ {"article_id": 29733, "qa_annotations": [...]}, ... ]
      2. {"articles": [ ... ]}
      3. {"answer_key": [ ... ]}
      4. {"29733": {"qa_annotations": [...]}, "29734": {...}}

    The current project answer key uses shape #1.
    """
    if isinstance(raw, list):
        return raw

    if not isinstance(raw, dict):
        raise ValueError("Answer-key JSON must be a list of article records or a dict containing them.")

    for container_key in ("articles", "answer_key", "items", "data"):
        if container_key in raw and isinstance(raw[container_key], list):
            return raw[container_key]

    # Dict keyed by article_id.
    records: List[Dict[str, Any]] = []
    for article_id, value in raw.items():
        if not isinstance(value, dict):
            continue
        item = dict(value)
        item.setdefault("article_id", int(article_id) if str(article_id).isdigit() else article_id)
        records.append(item)
    if records:
        return records

    raise ValueError(
        "Could not find article records in answer-key JSON. Expected a list, "
        "or a dict with key 'articles'/'answer_key', or a dict keyed by article_id."
    )


def _normalize_qa_annotation(q: Dict[str, Any], article_id: int, idx: int) -> Dict[str, Any]:
    """Normalize a QA record to the field names used by the scorer."""
    if not isinstance(q, dict):
        raise ValueError(f"Article {article_id} question {idx + 1}: QA annotation is not an object")

    question_text = q.get("question-text", q.get("question", q.get("question_text", "")))
    options = q.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError(f"Article {article_id} question {idx + 1}: missing or invalid options list")

    try:
        correct_option = int(q["correct_option"])
    except Exception as e:
        raise ValueError(f"Article {article_id} question {idx + 1}: missing/invalid correct_option") from e

    if correct_option < 1 or correct_option > len(options):
        raise ValueError(
            f"Article {article_id} question {idx + 1}: correct_option={correct_option} "
            f"is outside 1..{len(options)}"
        )

    question_in_set = int(q.get("question_in_set", idx + 1))
    correct_answer = q.get("correct_answer", options[correct_option - 1])

    out = dict(q)
    out["question_in_set"] = question_in_set
    out["question-text"] = question_text
    out["options"] = options
    out["correct_option"] = correct_option
    out["correct_answer"] = correct_answer
    out.setdefault("question_type", "multiple_choice" if len(options) > 3 else "true_false")
    return out


def load_answer_key(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load and validate answer-key JSON.

    The file should usually be named something like `answer_key.json`, but the
    parser does not require the `.json` extension so older `.txt` exports still
    work if they contain valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Answer-key JSON file not found: {path}")

    with path.open("r", encoding="utf-8", errors="replace") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Answer key must be valid JSON. Failed to parse {path} at "
                f"line {e.lineno}, column {e.colno}: {e.msg}"
            ) from e

    records = _coerce_answer_key_records(raw)
    key: Dict[int, Dict[str, Any]] = {}
    for item_idx, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Answer-key record {item_idx + 1} is not an object")
        if "article_id" not in item:
            raise ValueError(f"Answer-key record {item_idx + 1} is missing article_id")
        article_id = int(item["article_id"])
        qa = item.get("qa_annotations", item.get("questions"))
        if not isinstance(qa, list):
            raise ValueError(f"Article {article_id} is missing qa_annotations/questions list")
        if len(qa) != 6:
            raise ValueError(f"Article {article_id} has {len(qa)} questions; expected 6")

        normalized = dict(item)
        normalized["article_id"] = article_id
        normalized["qa_annotations"] = [
            _normalize_qa_annotation(q, article_id, idx) for idx, q in enumerate(qa)
        ]
        key[article_id] = normalized

    expected_article_ids = {article_id for rows in ORDER_MAP.values() for _, article_id, *_ in rows}
    missing = sorted(expected_article_ids - set(key))
    if missing:
        raise ValueError(f"Answer-key JSON is missing article_ids used in the study design: {missing}")
    return key


def load_qualtrics(path: Path, group: str, exclude_response_ids: Optional[set[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if df.shape[0] < 3:
        raise ValueError(f"{path} has too few rows for a Qualtrics export")
    # Drop Qualtrics display-label row and ImportId row.
    df = df.iloc[2:].copy().reset_index(drop=True)
    # Keep completed rows by default.
    if "Finished" in df.columns:
        df = df[df["Finished"].astype(str).str.strip().eq("1")].copy()
    if "Progress" in df.columns:
        df = df[pd.to_numeric(df["Progress"], errors="coerce").fillna(0) >= 100].copy()
    if exclude_response_ids:
        df = df[~df["ResponseId"].isin(exclude_response_ids)].copy()
    df.insert(0, "group", group)
    df.insert(1, "student_id", range(1, len(df) + 1))
    return df.reset_index(drop=True)


def build_human_long(
    g3a_path: Path,
    g3b_path: Path,
    answer_key_path: Path,
    exclude_response_ids: Optional[set[str]] = None,
    strict_counts: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    answer_key = load_answer_key(answer_key_path)
    warnings: List[str] = []
    frames = []
    for group, path in [("G3a", g3a_path), ("G3b", g3b_path)]:
        df = load_qualtrics(path, group, exclude_response_ids)
        observed = len(df)
        expected = EXPECTED_N[group]
        if observed != expected:
            msg = (
                f"WARNING: {group} has {observed} completed participant rows after filtering; "
                f"the design note says {expected}. No row was dropped automatically. "
                f"Use --exclude_response_ids if one response should be removed."
            )
            if strict_counts:
                raise ValueError(msg)
            warnings.append(msg)
        frames.append(df)

    rows: List[Dict[str, Any]] = []
    for df in frames:
        group = str(df["group"].iloc[0])
        qcols = get_question_columns_in_display_order(df)
        for _, resp in df.iterrows():
            response_id = resp.get("ResponseId", np.nan)
            prolific_id = resp.get("ProlificID", np.nan)
            student_id = int(resp["student_id"])
            for order, article_id, set_name, category, version_label in ORDER_MAP[group]:
                if article_id not in answer_key:
                    raise KeyError(f"Article {article_id} is missing from answer key")
                item = answer_key[article_id]
                article_key = article_key_from_label(version_label)
                qa_ann = item["qa_annotations"]
                if len(qa_ann) != 6:
                    raise ValueError(f"Article {article_id} has {len(qa_ann)} questions; expected 6")

                for q in qa_ann:
                    q_in_set = int(q["question_in_set"])
                    q_idx = q_in_set - 1
                    pre_col = get_q_column(qcols, qnum_for(order, q_in_set, "pre"))
                    post_col = get_q_column(qcols, qnum_for(order, q_in_set, "post"))
                    options = q["options"]
                    correct_option = int(q["correct_option"])
                    idk_option = len(options)
                    pre_ans = parse_answer(resp.get(pre_col))
                    post_ans = parse_answer(resp.get(post_col))
                    cls_pre = classify_answer(pre_ans, correct_option, idk_option)
                    cls_post = classify_answer(post_ans, correct_option, idk_option)
                    pre_correct = correct_flag(cls_pre)
                    post_correct = correct_flag(cls_post)
                    kgain_question = (
                        post_correct - pre_correct
                        if not pd.isna(post_correct) and not pd.isna(pre_correct)
                        else np.nan
                    )

                    rows.append(
                        {
                            "student_id": student_id,
                            "group": group,
                            "source_response_id": response_id,
                            "prolific_id": prolific_id,
                            "order": order,
                            "article_id": article_id,
                            "set": set_name,
                            "category": category,
                            "article_key": article_key,
                            "article_version_label": version_label,
                            "question_index": q_idx,
                            "question_in_set": q_in_set,
                            "question": q.get("question-text", q.get("question", "")),
                            "options": json.dumps(options, ensure_ascii=False),
                            "correct_option": correct_option,
                            "correct_answer": q["correct_answer"],
                            "idk_option": idk_option,
                            "human_pre_answer": int(pre_ans) if not pd.isna(pre_ans) else np.nan,
                            "human_post_answer": int(post_ans) if not pd.isna(post_ans) else np.nan,
                            "classification_pre": cls_pre,
                            "classification_post": cls_post,
                            "pre_correct": pre_correct,
                            "post_correct": post_correct,
                            "kgain_question": kgain_question,
                            "learned_from_idk": int(cls_pre == "dk" and cls_post == "correct"),
                            "corrected_misconception": int(cls_pre == "incorrect" and cls_post == "correct"),
                            "lost_correct": int(cls_pre == "correct" and cls_post != "correct"),
                            "pre_col": pre_col,
                            "post_col": post_col,
                        }
                    )

    human_long = pd.DataFrame(rows)
    return human_long, warnings


def aggregate_participant_article(human_long: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "group",
        "student_id",
        "source_response_id",
        "prolific_id",
        "order",
        "article_id",
        "set",
        "category",
        "article_key",
        "article_version_label",
    ]
    agg = (
        human_long.groupby(group_cols, dropna=False)
        .agg(
            pre_acc=("pre_correct", "mean"),
            post_acc=("post_correct", "mean"),
            kgain=("kgain_question", "mean"),
            pre_idk_rate=("classification_pre", lambda s: (s == "dk").mean()),
            post_idk_rate=("classification_post", lambda s: (s == "dk").mean()),
            learned_from_idk_n=("learned_from_idk", "sum"),
            corrected_misconception_n=("corrected_misconception", "sum"),
            lost_correct_n=("lost_correct", "sum"),
            n_questions=("question_index", "count"),
        )
        .reset_index()
    )
    agg["normalized_gain"] = np.where(
        agg["pre_acc"] < 1.0,
        (agg["post_acc"] - agg["pre_acc"]) / (1.0 - agg["pre_acc"]),
        np.nan,
    )
    return agg


def aggregate_question_item(human_long: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "article_id",
        "set",
        "category",
        "article_key",
        "article_version_label",
        "question_index",
        "question_in_set",
        "question",
        "correct_option",
        "correct_answer",
        "idk_option",
    ]
    return summarize_numeric(human_long, group_cols, ["pre_correct", "post_correct", "kgain_question"])


def build_summaries(participant_article: pd.DataFrame, question_item: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    article_summary = summarize_numeric(
        participant_article,
        ["article_id", "set", "category", "article_key", "article_version_label"],
        ["pre_acc", "post_acc", "kgain", "normalized_gain", "pre_idk_rate", "post_idk_rate"],
    ).sort_values(["article_id", "article_key"])

    system_summary = summarize_numeric(
        participant_article,
        ["article_key"],
        ["pre_acc", "post_acc", "kgain", "normalized_gain", "pre_idk_rate", "post_idk_rate"],
    ).sort_values("article_key")

    category_summary = summarize_numeric(
        participant_article,
        ["category", "article_key"],
        ["pre_acc", "post_acc", "kgain", "normalized_gain"],
    ).sort_values(["category", "article_key"])

    # Topic-level paired comparison: each article_id has a mean KGain for news_0 and news_2.
    pivot = article_summary.pivot_table(
        index=["article_id", "category"],
        columns="article_key",
        values=["pre_acc_mean", "post_acc_mean", "kgain_mean", "normalized_gain_mean"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{key}" for metric, key in pivot.columns]
    paired = pivot.reset_index()
    if {"kgain_mean_news_0", "kgain_mean_news_2"}.issubset(paired.columns):
        paired["kgain_diff_news_0_minus_news_2"] = paired["kgain_mean_news_0"] - paired["kgain_mean_news_2"]
    if {"post_acc_mean_news_0", "post_acc_mean_news_2"}.issubset(paired.columns):
        paired["post_acc_diff_news_0_minus_news_2"] = paired["post_acc_mean_news_0"] - paired["post_acc_mean_news_2"]

    return {
        "article_summary": article_summary,
        "system_summary": system_summary,
        "category_summary": category_summary,
        "paired_article_comparison": paired,
        "question_item_summary": question_item,
    }


# LLMSim correlation helpers
def read_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return pd.DataFrame(records)


def normalize_llmsim(llm: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "student_id",
        "group",
        "article_id",
        "article_key",
        "article_version_label",
        "question_index",
        "classification_pre",
        "classification_post",
    ]
    missing = [c for c in needed if c not in llm.columns]
    if missing:
        raise ValueError(f"LLMSim JSONL is missing required fields: {missing}")
    out = llm.copy()
    out["pre_correct"] = (out["classification_pre"] == "correct").astype(float)
    out["post_correct"] = (out["classification_post"] == "correct").astype(float)
    out["kgain_question"] = out["post_correct"] - out["pre_correct"]
    out["pre_idk"] = (out["classification_pre"] == "dk").astype(float)
    out["post_idk"] = (out["classification_post"] == "dk").astype(float)
    return out


def aggregate_for_correlation(df: pd.DataFrame, source: str, level: str) -> pd.DataFrame:
    if level == "article":
        keys = ["article_id", "article_key", "article_version_label"]
    elif level == "article_question":
        keys = ["article_id", "article_key", "article_version_label", "question_index"]
    elif level == "article_key":
        keys = ["article_key"]
    else:
        raise ValueError("level must be article, article_question, or article_key")
    out = (
        df.groupby(keys, dropna=False)
        .agg(
            pre_acc=("pre_correct", "mean"),
            post_acc=("post_correct", "mean"),
            kgain=("kgain_question", "mean"),
            pre_idk_rate=(
                "pre_idk" if "pre_idk" in df.columns else "classification_pre",
                lambda s: (s == "dk").mean() if s.dtype == object else s.mean(),
            ),
            post_idk_rate=(
                "post_idk" if "post_idk" in df.columns else "classification_post",
                lambda s: (s == "dk").mean() if s.dtype == object else s.mean(),
            ),
            n=("kgain_question", "count"),
        )
        .reset_index()
    )
    return out.rename(
        columns={
            "pre_acc": f"{source}_pre_acc",
            "post_acc": f"{source}_post_acc",
            "kgain": f"{source}_kgain",
            "pre_idk_rate": f"{source}_pre_idk_rate",
            "post_idk_rate": f"{source}_post_idk_rate",
            "n": f"{source}_n",
        }
    )


def corr_pair(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(data)
    if n < 3:
        return {"n": n, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    if stats is not None:
        pr, pp = stats.pearsonr(data["x"], data["y"])
        sr, sp = stats.spearmanr(data["x"], data["y"])
        return {"n": n, "pearson_r": pr, "pearson_p": pp, "spearman_r": sr, "spearman_p": sp}
    return {
        "n": n,
        "pearson_r": data["x"].corr(data["y"], method="pearson"),
        "pearson_p": np.nan,
        "spearman_r": data["x"].corr(data["y"], method="spearman"),
        "spearman_p": np.nan,
    }


def compute_llmsim_correlations(human_long: pd.DataFrame, llmsim_path: Path, outdir: Path) -> pd.DataFrame:
    llm = normalize_llmsim(read_jsonl(llmsim_path))
    records = []
    for level in ["article", "article_question", "article_key"]:
        human_agg = aggregate_for_correlation(human_long, "human", level)
        llm_agg = aggregate_for_correlation(llm, "llmsim", level)
        keys = [
            c
            for c in human_agg.columns
            if c in llm_agg.columns
            and c in {"article_id", "article_key", "article_version_label", "question_index"}
        ]
        merged = human_agg.merge(llm_agg, on=keys, how="inner")
        write_csv(merged, outdir / f"human_llmsim_merged_{level}.csv")
        for metric in ["pre_acc", "post_acc", "kgain", "pre_idk_rate", "post_idk_rate"]:
            res = corr_pair(merged[f"human_{metric}"], merged[f"llmsim_{metric}"])
            records.append({"level": level, "metric": metric, **res})
    corr = pd.DataFrame(records)
    write_csv(corr, outdir / "human_llmsim_correlations.csv")
    return corr


# Plots and stats
def plot_boxplots(participant_article: pd.DataFrame, outdir: Path) -> None:
    long = participant_article.melt(
        id_vars=["group", "student_id", "article_id", "article_key", "article_version_label"],
        value_vars=["pre_acc", "post_acc", "kgain"],
        var_name="measure",
        value_name="score",
    )
    labels = ["pre_acc", "post_acc", "kgain"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = []
    tick_labels = []
    positions = []
    pos = 1
    for measure in labels:
        for key in ["news_0", "news_2"]:
            vals = long[(long["measure"] == measure) & (long["article_key"] == key)]["score"].dropna()
            data.append(vals.values)
            tick_labels.append(f"{measure}\n{key}")
            positions.append(pos)
            pos += 1
        pos += 0.5
    ax.boxplot(data, positions=positions, showmeans=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_ylabel("Score / fraction correct")
    ax.set_title("Human pre, post, and KnowledgeGain by article system")
    ax.axhline(0, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(outdir / "boxplot_pre_post_kgain_by_system.png", dpi=200)
    plt.close(fig)


def plot_paired_article_kgain(paired: pd.DataFrame, outdir: Path) -> None:
    needed = {"article_id", "kgain_mean_news_0", "kgain_mean_news_2"}
    if not needed.issubset(paired.columns):
        return
    p = paired.sort_values("article_id").reset_index(drop=True)
    x = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, p["kgain_mean_news_0"], marker="o", label="news_0")
    ax.plot(x, p["kgain_mean_news_2"], marker="o", label="news_2")
    for i, row in p.iterrows():
        ax.plot([i, i], [row["kgain_mean_news_0"], row["kgain_mean_news_2"]], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(p["article_id"].astype(str), rotation=90)
    ax.set_ylabel("Mean KGain per article")
    ax.set_title("Topic-paired human KGain: news_0 vs news_2")
    ax.legend()
    ax.axhline(0, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(outdir / "paired_article_kgain_news0_vs_news2.png", dpi=200)
    plt.close(fig)


def run_system_tests(paired: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for metric in ["pre_acc_mean", "post_acc_mean", "kgain_mean", "normalized_gain_mean"]:
        c0 = f"{metric}_news_0"
        c2 = f"{metric}_news_2"
        if c0 not in paired.columns or c2 not in paired.columns:
            continue
        d = paired[[c0, c2]].dropna()
        rec: Dict[str, Any] = {
            "metric": metric,
            "n_articles": len(d),
            "news_0_mean": d[c0].mean(),
            "news_2_mean": d[c2].mean(),
            "mean_diff_news_0_minus_news_2": (d[c0] - d[c2]).mean(),
        }
        if len(d) >= 2 and stats is not None:
            try:
                t = stats.ttest_rel(d[c0], d[c2])
                rec["paired_t_stat"] = t.statistic
                rec["paired_t_p"] = t.pvalue
            except Exception:
                rec["paired_t_stat"] = np.nan
                rec["paired_t_p"] = np.nan
            try:
                w = stats.wilcoxon(d[c0], d[c2], zero_method="wilcox")
                rec["wilcoxon_stat"] = w.statistic
                rec["wilcoxon_p"] = w.pvalue
            except Exception:
                rec["wilcoxon_stat"] = np.nan
                rec["wilcoxon_p"] = np.nan
        records.append(rec)
    tests = pd.DataFrame(records)
    write_csv(tests, outdir / "system_paired_tests.csv")
    return tests


# Main
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze human KGain from G3a/G3b Qualtrics exports.")
    ap.add_argument("--g3a", type=Path, default=Path("g3a.csv"))
    ap.add_argument("--g3b", type=Path, default=Path("g3b.csv"))
    ap.add_argument(
        "--answer_key_json",
        "--answer_key",
        dest="answer_key",
        type=Path,
        default=Path("answer_key.json"),
        help="Path to answer-key JSON. The older --answer_key name is kept as an alias.",
    )
    ap.add_argument("--llmsim_jsonl", type=Path, default=None, help="Optional LLMSim predictions JSONL.")
    ap.add_argument("--outdir", type=Path, default=Path("kgain_outputs"))
    ap.add_argument(
        "--exclude_response_ids",
        type=str,
        default="",
        help="Comma-separated Qualtrics ResponseIds to exclude, for manual QC.",
    )
    ap.add_argument(
        "--strict_counts",
        action="store_true",
        help="Raise an error if completed rows do not match expected G3a=5/G3b=6.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    exclude = {x.strip() for x in args.exclude_response_ids.split(",") if x.strip()} or None

    human_long, warnings = build_human_long(
        g3a_path=args.g3a,
        g3b_path=args.g3b,
        answer_key_path=args.answer_key,
        exclude_response_ids=exclude,
        strict_counts=args.strict_counts,
    )
    # Add numeric id that is unique across groups for easier mixed-model/correlation use.
    human_long["human_reader_uid"] = human_long["group"] + "_" + human_long["student_id"].astype(str)

    # Build long phase-level binary correctness table for mixed-effects logistic regression.
    # This is raw binary correctness, not normalized gain. It is kept for comparison.
    pre = human_long.copy()
    pre["phase"] = "pre"
    pre["correct"] = pre["pre_correct"]

    post = human_long.copy()
    post["phase"] = "post"
    post["correct"] = post["post_correct"]

    gen_long = pd.concat([pre, post], ignore_index=True)

    # Rename / construct columns for the binary logistic model.
    gen_long["participant"] = gen_long["human_reader_uid"]
    gen_long["topic"] = gen_long["article_id"].astype(str)
    gen_long["question_uid"] = (
        gen_long["article_id"].astype(str)
        + "_q"
        + gen_long["question_in_set"].astype(str)
    )

    # Confirmed mapping for the generated-article study:
    # news_2 = Baseline, news_0 = Ours.
    gen_long["system"] = gen_long["article_key"].map(
        {
            "news_0": "Ours",
            "news_2": "Baseline",
        }
    )

    gen_long = gen_long.dropna(subset=["correct", "system"])
    gen_long["correct"] = gen_long["correct"].astype(int)

    cols = [
        "participant",
        "topic",
        "question_uid",
        "system",
        "phase",
        "correct",
        "article_id",
        "question_in_set",
        "article_key",
    ]

    gen_long_path = args.outdir / "generated_kgain_correct_long.csv"
    write_csv(gen_long[cols], gen_long_path)

    print(gen_long[cols].head())
    print(gen_long.groupby(["system", "phase"])["correct"].mean())
    print(f"Wrote generated long binary CSV to: {gen_long_path.resolve()}")

    # Preserve a JSONL option that mirrors LLMSim row layout as closely as possible.
    write_csv(human_long, args.outdir / "human_question_level_annotated.csv")
    with (args.outdir / "human_question_level_annotated.jsonl").open("w", encoding="utf-8") as f:
        for rec in human_long.to_dict(orient="records"):
            # Convert NumPy NaN to None for valid JSON.
            clean = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
            # Decode options back into arrays for JSONL compatibility.
            if isinstance(clean.get("options"), str):
                clean["options"] = json.loads(clean["options"])
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    participant_article = aggregate_participant_article(human_long)
    write_csv(participant_article, args.outdir / "participant_article_scores.csv")

    # Build participant/article-level normalized KGAIN table.
    # Normalized KGAIN is defined on pre/post proportions, not on individual
    # binary question rows:
    #     normalized_gain = (post_acc - pre_acc) / (1 - pre_acc)
    # Rows with pre_acc == 1 are undefined and are dropped from the model table.
    norm_gain = participant_article.copy()
    norm_gain["human_reader_uid"] = norm_gain["group"] + "_" + norm_gain["student_id"].astype(str)
    norm_gain["participant"] = norm_gain["human_reader_uid"]
    norm_gain["topic"] = norm_gain["article_id"].astype(str)
    norm_gain["system"] = norm_gain["article_key"].map(
        {
            "news_0": "Ours",
            "news_2": "Baseline",
        }
    )
    norm_gain = norm_gain.dropna(subset=["normalized_gain", "system"])

    norm_cols = [
        "participant",
        "topic",
        "system",
        "normalized_gain",
        "kgain",
        "pre_acc",
        "post_acc",
        "article_id",
        "article_key",
        "article_version_label",
        "group",
        "student_id",
        "order",
        "category",
        "n_questions",
    ]

    norm_gain_path = args.outdir / "generated_kgain_normalized.csv"
    write_csv(norm_gain[norm_cols], norm_gain_path)
    print(norm_gain[norm_cols].head())
    print(norm_gain.groupby("system")["normalized_gain"].mean())
    print(f"Wrote generated normalized KGAIN CSV to: {norm_gain_path.resolve()}")

    question_item = aggregate_question_item(human_long)
    summaries = build_summaries(participant_article, question_item)
    for name, df in summaries.items():
        write_csv(df, args.outdir / f"{name}.csv")

    plot_boxplots(participant_article, args.outdir)
    plot_paired_article_kgain(summaries["paired_article_comparison"], args.outdir)
    tests = run_system_tests(summaries["paired_article_comparison"], args.outdir)

    corr = None
    if args.llmsim_jsonl is not None:
        corr = compute_llmsim_correlations(human_long, args.llmsim_jsonl, args.outdir)

    # Console summary for quick sanity-checking.
    print("\nHuman KGain analysis complete")
    print("=" * 35)
    for msg in warnings:
        print(msg, file=sys.stderr)
    print(f"Question-level rows: {len(human_long):,}")
    print(f"Participant/article rows: {len(participant_article):,}")
    print("\nSystem summary:")
    cols = [
        "article_key",
        "pre_acc_mean",
        "post_acc_mean",
        "kgain_mean",
        "normalized_gain_mean",
        "kgain_n",
    ]
    print(summaries["system_summary"][cols].to_string(index=False))
    print("\nPaired topic-level tests:")
    print(tests.to_string(index=False))
    if corr is not None:
        print("\nHuman vs LLMSim correlations:")
        print(corr.to_string(index=False))
    print(f"\nWrote outputs to: {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
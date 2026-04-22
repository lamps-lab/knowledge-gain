#!/usr/bin/env python3
# scripts/llmsim_condition_alignment.py

"""
python scripts/llmsim_condition_alignment.py \
  --dataset ../data/kgain_annotated_dataset.json \
  --predictions runs/llmsim_ablation/04_verbalized_sampling/predictions.jsonl \
  --out_dir runs/llmsim_condition_alignment_verbalized/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


IDK_TEXT = "I do not know the answer."
EPS = 1e-9


def safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def find_idk(options: List[str]) -> int:
    try:
        return options.index(IDK_TEXT) + 1
    except ValueError:
        return len(options)


def bucket(ans: int, correct: Optional[int], idk: int) -> int:
    """
    0 = correct, 1 = incorrect, 2 = IDK
    """
    if ans == idk:
        return 2
    if correct is not None and ans == correct:
        return 0
    return 1


def raw_dist(counts: List[int]) -> List[float]:
    s = sum(counts)
    if s <= 0:
        return [float("nan")] * len(counts)
    return [c / s for c in counts]


def smooth_dist(counts: List[int], alpha: float = 1e-3) -> List[float]:
    xs = [float(c) + alpha for c in counts]
    s = sum(xs)
    return [x / s for x in xs]


def kl_div(p: List[float], q: List[float]) -> float:
    return sum(
        max(EPS, pi) * math.log(max(EPS, pi) / max(EPS, qi))
        for pi, qi in zip(p, q)
    )


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(xs: List[float], ys: List[float]) -> float:
    pairs = [
        (x, y)
        for x, y in zip(xs, ys)
        if math.isfinite(x) and math.isfinite(y)
    ]

    if len(pairs) < 2:
        return float("nan")

    xs, ys = zip(*pairs)
    mx, my = mean(list(xs)), mean(list(ys))

    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)

    if vx <= 0 or vy <= 0:
        return float("nan")

    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def fmt(x: float, digits: int = 3) -> str:
    if x is None or not math.isfinite(float(x)):
        return "--"
    return f"{float(x):.{digits}f}"


def texesc(s: str) -> str:
    return (
        s.replace("&", r"\&")
        .replace("_", r"\_")
        .replace("%", r"\%")
    )


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_human(dataset_path: str):
    """
    Returns:
      meta:
        key = (doc_id, question_index, annotator_id)
        value = {
          media, doc_key, condition_pre_key, condition_post_key,
          correct, idk
        }

      human_condition_counts:
        key = condition, one of PRE/news/abstract/tweet
        value = [correct, incorrect, idk]

      human_unit_counts:
        key = (condition, doc_id, question_index)
        value = [correct, incorrect, idk]

      human_pre_correct:
        key = (media, doc_id)
        value = list of pre correctness indicators

      human_post_correct:
        key = (media, doc_id)
        value = list of post correctness indicators
    """
    raw = json.load(open(dataset_path, encoding="utf-8"))

    meta = {}

    human_condition_counts = defaultdict(lambda: [0, 0, 0])
    human_unit_counts = defaultdict(lambda: [0, 0, 0])

    human_pre_correct = defaultdict(list)
    human_post_correct = defaultdict(list)

    for doc_id, doc in enumerate(raw):
        media = doc["content-type"]
        anns = doc.get("human_annotations") or []

        if not anns:
            continue

        for q_idx, qref in enumerate(anns[0]["qa_annotations"]):
            options = qref["options"]
            idk = find_idk(options)

            correct = qref.get("correct_option")
            correct = int(correct) if correct is not None else None

            for ann in anns:
                annotator_id = int(ann["annotator_id"])
                qa = ann["qa_annotations"][q_idx]

                key = (doc_id, q_idx, annotator_id)
                doc_key = (media, doc_id)

                meta[key] = {
                    "media": media,
                    "doc_key": doc_key,
                    "correct": correct,
                    "idk": idk,
                }

                pre_ans = safe_int(qa.get("human-answer-pre"))
                post_ans = safe_int(qa.get("human-answer-post"))

                pre_bucket = bucket(pre_ans, correct, idk)
                human_condition_counts["PRE"][pre_bucket] += 1
                human_unit_counts[("PRE", doc_id, q_idx)][pre_bucket] += 1

                human_pre_correct[doc_key].append(
                    int(correct is not None and pre_ans == correct)
                )

                if media in ("news", "abstract", "tweet") and qa.get("human-answer-post") is not None:
                    post_bucket = bucket(post_ans, correct, idk)
                    human_condition_counts[media][post_bucket] += 1
                    human_unit_counts[(media, doc_id, q_idx)][post_bucket] += 1

                    human_post_correct[doc_key].append(
                        int(correct is not None and post_ans == correct)
                    )

    human_kg = {}
    for doc_key in set(human_pre_correct) & set(human_post_correct):
        if human_pre_correct[doc_key] and human_post_correct[doc_key]:
            human_kg[doc_key] = mean(human_post_correct[doc_key]) - mean(human_pre_correct[doc_key])

    return (
        meta,
        human_condition_counts,
        human_unit_counts,
        human_pre_correct,
        human_post_correct,
        human_kg,
    )


def load_model_predictions(pred_path: str, meta: Dict[Tuple[int, int, int], dict]):
    """
    Returns model counts and KG values in the same format as human data.
    """
    model_condition_counts = defaultdict(lambda: [0, 0, 0])
    model_unit_counts = defaultdict(lambda: [0, 0, 0])

    model_pre_correct = defaultdict(list)
    model_post_correct = defaultdict(list)

    seen = set()

    for r in load_jsonl(pred_path):
        doc_id = safe_int(r.get("doc_id"))
        q_idx = safe_int(r.get("question_index"))
        annotator_id = safe_int(r.get("annotator_id"))

        key = (doc_id, q_idx, annotator_id)

        if key not in meta or key in seen:
            continue

        seen.add(key)

        m = meta[key]
        media = m["media"]
        doc_key = m["doc_key"]
        correct = m["correct"]
        idk = m["idk"]

        pre_ans = safe_int(r.get("model_pre_answer"))
        post_ans = safe_int(r.get("model_post_answer"))

        pre_bucket = bucket(pre_ans, correct, idk)
        model_condition_counts["PRE"][pre_bucket] += 1
        model_unit_counts[("PRE", doc_id, q_idx)][pre_bucket] += 1

        model_pre_correct[doc_key].append(
            int(correct is not None and pre_ans == correct)
        )

        if media in ("news", "abstract", "tweet"):
            post_bucket = bucket(post_ans, correct, idk)
            model_condition_counts[media][post_bucket] += 1
            model_unit_counts[(media, doc_id, q_idx)][post_bucket] += 1

            model_post_correct[doc_key].append(
                int(correct is not None and post_ans == correct)
            )

    model_kg = {}
    for doc_key in set(model_pre_correct) & set(model_post_correct):
        if model_pre_correct[doc_key] and model_post_correct[doc_key]:
            model_kg[doc_key] = mean(model_post_correct[doc_key]) - mean(model_pre_correct[doc_key])

    return model_condition_counts, model_unit_counts, model_kg


def evaluate_condition_alignment(
    human_condition_counts,
    human_unit_counts,
    human_kg,
    model_condition_counts,
    model_unit_counts,
    model_kg,
):
    rows = []

    for condition in ["PRE", "news", "abstract", "tweet"]:
        hc = human_condition_counts.get(condition, [0, 0, 0])
        mc = model_condition_counts.get(condition, [0, 0, 0])

        if sum(hc) == 0 or sum(mc) == 0:
            global_kl = float("nan")
            correct_mae = float("nan")
            idk_mae = float("nan")
        else:
            hp = raw_dist(hc)
            mp = raw_dist(mc)
            global_kl = kl_div(hp, mp)
            correct_mae = abs(hp[0] - mp[0])
            idk_mae = abs(hp[2] - mp[2])

        item_kls = []
        item_correct_maes = []
        item_idk_maes = []

        for unit_key, huc in human_unit_counts.items():
            unit_condition = unit_key[0]
            if unit_condition != condition:
                continue

            muc = model_unit_counts.get(unit_key)
            if not muc or sum(huc) == 0 or sum(muc) == 0:
                continue

            hp_s = smooth_dist(huc)
            mp_s = smooth_dist(muc)
            item_kls.append(kl_div(hp_s, mp_s))

            hp = raw_dist(huc)
            mp = raw_dist(muc)
            item_correct_maes.append(abs(hp[0] - mp[0]))
            item_idk_maes.append(abs(hp[2] - mp[2]))

        if condition == "PRE":
            kg_corr = float("nan")
        else:
            docs = sorted(
                d for d in set(human_kg) & set(model_kg)
                if d[0] == condition
            )
            kg_corr = pearson(
                [human_kg[d] for d in docs],
                [model_kg[d] for d in docs],
            )

        rows.append({
            "condition": condition,
            "global_kl": global_kl,
            "item_kl": mean(item_kls),
            "correct_mae": correct_mae,
            "item_correct_mae": mean(item_correct_maes),
            "kg_corr": kg_corr,
            "idk_mae": idk_mae,
            "item_idk_mae": mean(item_idk_maes),
            "human_correct": raw_dist(hc)[0] if sum(hc) else float("nan"),
            "model_correct": raw_dist(mc)[0] if sum(mc) else float("nan"),
            "human_incorrect": raw_dist(hc)[1] if sum(hc) else float("nan"),
            "model_incorrect": raw_dist(mc)[1] if sum(mc) else float("nan"),
            "human_idk": raw_dist(hc)[2] if sum(hc) else float("nan"),
            "model_idk": raw_dist(mc)[2] if sum(mc) else float("nan"),
        })

    return rows


def write_csv(rows: List[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def condition_label(c: str) -> str:
    return {
        "PRE": "PRE",
        "news": "News",
        "abstract": "Abstract",
        "tweet": "Tweet",
    }.get(c, c)


def write_latex_main(rows: List[dict], path: str):
    """
    Main-paper table: compact condition-level alignment.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    head = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
Condition & KL $\downarrow$ & Correct MAE $\downarrow$ & KG corr. $\uparrow$ & IDK MAE $\downarrow$ \\
\midrule
"""

    body = ""
    for r in rows:
        body += (
            f"{condition_label(r['condition'])} "
            f"& {fmt(r['global_kl'])} "
            f"& {fmt(r['correct_mae'])} "
            f"& {fmt(r['kg_corr'])} "
            f"& {fmt(r['idk_mae'])} \\\\\n"
        )

    tail = r"""\bottomrule
\end{tabular}
\caption{Alignment between human and full {\sc LLMSim} answer-outcome distributions by condition. KL measures aggregate alignment over Correct/Incorrect/IDK outcomes. Correct MAE and IDK MAE measure calibration of correctness and abstention rates. KG correlation is computed over document-level pre/post gains and is not defined for PRE alone.}
\label{tab:llmsim_condition_alignment}
\end{table}
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(head + body + tail)


def write_latex_appendix(rows: List[dict], path: str):
    """
    Appendix table with aggregate human/model distributions.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    head = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrrr}
\toprule
Condition & Human C & Model C & Human I & Model I & Human IDK & Model IDK \\
\midrule
"""

    body = ""
    for r in rows:
        body += (
            f"{condition_label(r['condition'])} "
            f"& {fmt(100 * r['human_correct'], 1)} "
            f"& {fmt(100 * r['model_correct'], 1)} "
            f"& {fmt(100 * r['human_incorrect'], 1)} "
            f"& {fmt(100 * r['model_incorrect'], 1)} "
            f"& {fmt(100 * r['human_idk'], 1)} "
            f"& {fmt(100 * r['model_idk'], 1)} \\\\\n"
        )

    tail = r"""\bottomrule
\end{tabular}
\caption{Aggregate human and full {\sc LLMSim} answer-outcome distributions by condition. Values are percentages. C denotes Correct and I denotes Incorrect.}
\label{tab:llmsim_condition_distributions}
\end{table}
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(head + body + tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Human annotated dataset JSON.")
    ap.add_argument("--predictions", required=True, help="LLMSim predictions JSONL.")
    ap.add_argument("--out_dir", default="runs/llmsim_condition_alignment/tables")
    args = ap.parse_args()

    (
        meta,
        human_condition_counts,
        human_unit_counts,
        human_pre_correct,
        human_post_correct,
        human_kg,
    ) = load_human(args.dataset)

    model_condition_counts, model_unit_counts, model_kg = load_model_predictions(
        args.predictions,
        meta,
    )

    rows = evaluate_condition_alignment(
        human_condition_counts,
        human_unit_counts,
        human_kg,
        model_condition_counts,
        model_unit_counts,
        model_kg,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, "llmsim_condition_alignment.csv")
    tex_path = os.path.join(args.out_dir, "llmsim_condition_alignment.tex")
    app_tex_path = os.path.join(args.out_dir, "llmsim_condition_distributions.tex")

    write_csv(rows, csv_path)
    write_latex_main(rows, tex_path)
    write_latex_appendix(rows, app_tex_path)

    print("\nCondition-level alignment:")
    for r in rows:
        print(
            f"{condition_label(r['condition'])}: "
            f"KL={fmt(r['global_kl'])}, "
            f"Correct MAE={fmt(r['correct_mae'])}, "
            f"KG corr={fmt(r['kg_corr'])}, "
            f"IDK MAE={fmt(r['idk_mae'])}"
        )
        print(
            f"  Human vs model aggregate: "
            f"Correct {fmt(100 * r['human_correct'], 1)}% vs {fmt(100 * r['model_correct'], 1)}%, "
            f"Incorrect {fmt(100 * r['human_incorrect'], 1)}% vs {fmt(100 * r['model_incorrect'], 1)}%, "
            f"IDK {fmt(100 * r['human_idk'], 1)}% vs {fmt(100 * r['model_idk'], 1)}%"
        )

    print("\nWrote:")
    print(f"  CSV:       {csv_path}")
    print(f"  Main TeX:  {tex_path}")
    print(f"  App. TeX:  {app_tex_path}")


if __name__ == "__main__":
    main()
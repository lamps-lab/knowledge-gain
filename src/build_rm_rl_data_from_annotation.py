#!/usr/bin/env python3
"""
build_rm_and_rl_data_from_annotated.py

Creates:
1) Reward-model preference dataset (JSONL) from HUMAN KGain in the annotated dataset.
2) RL training records (JSONL) containing abstracts + canonical question sets.

We DO NOT generate any new "candidates" with a model.
We ONLY use the existing annotated content items (news/abstract/tweet) per article-id.

Input format (your kgain_annotated_dataset.json):
[
  {
    "article-id": "...",
    "content-type": "abstract|news|tweet",
    "content": "...",
    "human_annotations": [
      {
        "annotator_id": ...,
        "qa_annotations": [
          {
            "question_in_set": 1,
            "question-text": "...",
            "options": [...],
            "correct_option": 2,
            "correct_answer": "...",
            "human-answer-pre": 3,
            "human-answer-post": 1
          }, ...
        ]
      }, ...
    ]
  }, ...
]

Outputs:
- rm_pairs.jsonl lines:
  {
    "article_id": "...",
    "prompt": "...",      # abstract + question set (no correct answers by default)
    "chosen": "...",      # higher-KGain content
    "rejected": "...",    # lower-KGain content
    "meta": {...}         # kg values etc
  }

- rl_records.jsonl lines:
  {
    "article_id": "...",
    "paper_abstract": "...",
    "qa_annotations": [...canonical...]
  }

Recommended usage:
  python build_rm_and_rl_data_from_annotated.py \
    --input kgain_annotated_dataset.json \
    --rm_output rm_pairs.jsonl \
    --rl_output rl_records.jsonl \
    --pairs_mode anchor_news \
    --min_margin 0.02
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple, Optional


IDK_TEXT = "I do not know the answer."


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


def canonicalize_questions(all_human_annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build canonical QA set per question_in_set by taking the most common (qtext, options, correct_option).
    """
    per_q_counter: Dict[int, Counter] = defaultdict(Counter)
    exemplars: Dict[Tuple[str, Tuple[str, ...], int], Dict[str, Any]] = {}

    for ann in all_human_annotations:
        for qa in (ann.get("qa_annotations") or []):
            qid = qa.get("question_in_set")
            if qid is None:
                continue
            try:
                qid = int(qid)
            except Exception:
                continue

            qtext = (qa.get("question-text") or "").strip()
            options = qa.get("options") or []
            try:
                correct = int(qa.get("correct_option"))
            except Exception:
                correct = 0

            key = (qtext, tuple(options), correct)
            per_q_counter[qid][key] += 1
            if key not in exemplars:
                exemplars[key] = {
                    "question_in_set": qid,
                    "question-text": qtext,
                    "options": list(options),
                    "correct_option": correct,
                    "correct_answer": qa.get("correct_answer", ""),
                }

    out: List[Dict[str, Any]] = []
    for qid in sorted(per_q_counter.keys()):
        key, _ = per_q_counter[qid].most_common(1)[0]
        row = exemplars[key]
        # Ensure IDK option exists; many of your items already include it
        opts = row.get("options") or []
        if opts and all((o or "").strip() != IDK_TEXT for o in opts):
            row["options"] = opts + [IDK_TEXT]
        out.append(row)
    return out


def compute_human_kgain_for_item(
    item_human_annotations: List[Dict[str, Any]],
    canonical_qas: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """
    Computes mean pre_acc, post_acc, kgain over annotators, using canonical correct options.
    """
    correct_by_qid = {}
    for q in canonical_qas:
        try:
            correct_by_qid[int(q["question_in_set"])] = int(q["correct_option"])
        except Exception:
            pass

    pre_accs: List[float] = []
    post_accs: List[float] = []
    kgains: List[float] = []

    for ann in item_human_annotations or []:
        pre_hits: List[float] = []
        post_hits: List[float] = []
        for qa in (ann.get("qa_annotations") or []):
            qid = qa.get("question_in_set")
            if qid is None:
                continue
            try:
                qid = int(qid)
            except Exception:
                continue
            correct = correct_by_qid.get(qid)
            if not correct:
                continue

            a_pre = qa.get("human-answer-pre")
            a_post = qa.get("human-answer-post")
            try:
                a_pre_i = int(a_pre)
            except Exception:
                a_pre_i = None
            try:
                a_post_i = int(a_post) if a_post is not None else None
            except Exception:
                a_post_i = None

            if a_pre_i is not None:
                pre_hits.append(1.0 if a_pre_i == correct else 0.0)
            if a_post_i is not None:
                post_hits.append(1.0 if a_post_i == correct else 0.0)

        if pre_hits:
            pre_acc = sum(pre_hits) / len(pre_hits)
            pre_accs.append(pre_acc)
        else:
            continue

        if post_hits:
            post_acc = sum(post_hits) / len(post_hits)
            post_accs.append(post_acc)
            kgains.append(post_acc - pre_acc)

    pre_mean = (sum(pre_accs) / len(pre_accs)) if pre_accs else None
    post_mean = (sum(post_accs) / len(post_accs)) if post_accs else None
    kg_mean = (sum(kgains) / len(kgains)) if kgains else None

    return {
        "pre_acc": pre_mean,
        "post_acc": post_mean,
        "kgain": kg_mean,
        "n_annotators_used": len(kgains),
    }


def qas_to_text(qas: List[Dict[str, Any]], include_correct: bool = False) -> str:
    blocks = []
    for qa in qas:
        qid = qa.get("question_in_set")
        qtext = qa.get("question-text") or ""
        options = qa.get("options") or []
        opt_lines = "\n".join([f"  {i+1}. {o}" for i, o in enumerate(options)])
        if include_correct:
            blocks.append(f"Q{qid}: {qtext}\n{opt_lines}\nCorrect: {qa.get('correct_option')}")
        else:
            blocks.append(f"Q{qid}: {qtext}\n{opt_lines}")
    return "\n\n".join(blocks)


def build_rm_prompt(abstract: str, qas: List[Dict[str, Any]], include_correct: bool) -> str:
    # This prompt is for the REWARD MODEL, not for the generator.
    # Generator should NOT see the questions to avoid “writing to the test”.
    return (
        "You are scoring a candidate science news article.\n"
        "Score higher if the candidate would increase a typical reader's ability to answer the questions after reading.\n\n"
        f"PAPER ABSTRACT:\n{abstract.strip()}\n\n"
        f"QUESTION SET:\n{qas_to_text(qas, include_correct=include_correct)}\n\n"
        "CANDIDATE CONTENT:\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="kgain_annotated_dataset.json")
    ap.add_argument("--rm_output", required=True, help="output JSONL for RewardTrainer (prompt/chosen/rejected)")
    ap.add_argument("--rl_output", required=True, help="output JSONL for RL records (abstract + qa set)")
    ap.add_argument(
        "--pairs_mode",
        choices=["all_ordered", "best_vs_worst", "anchor_news"],
        default="anchor_news",
        help=(
            "all_ordered: emit every ordered pair where kgain_i > kgain_j by margin\n"
            "best_vs_worst: only top vs bottom per article\n"
            "anchor_news: pairs are (news vs others) when news exists"
        ),
    )
    ap.add_argument("--min_margin", type=float, default=0.02, help="minimum kgain difference to create a preference pair")
    ap.add_argument("--include_correct_in_prompt", action="store_true", help="include correct option index in RM prompt")
    args = ap.parse_args()

    raw = load_json(args.input)
    # Group: article_id -> content_type -> record
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for doc in raw:
        aid = str(doc.get("article-id"))
        ctype = str(doc.get("content-type"))
        grouped[aid][ctype] = doc

    rm_rows: List[Dict[str, Any]] = []
    rl_rows: List[Dict[str, Any]] = []

    for aid, items in grouped.items():
        if "abstract" not in items:
            continue

        abstract_text = (items["abstract"].get("content") or "").strip()
        if not abstract_text:
            continue

        # Collect ALL human annotations from all content-types to canonicalize Qs robustly
        all_anns = []
        for it in items.values():
            all_anns.extend(it.get("human_annotations") or [])

        canonical_qas = canonicalize_questions(all_anns)
        if not canonical_qas:
            continue

        # RL record (for GRPO training): abstract + canonical Q set (generator sees only abstract; RM sees both)
        rl_rows.append({
            "article_id": aid,
            "paper_abstract": abstract_text,
            "qa_annotations": canonical_qas,
        })

        # Compute human kgain per available content type
        scores: Dict[str, Dict[str, Any]] = {}
        texts: Dict[str, str] = {}
        for ctype in ["news", "abstract", "tweet"]:
            if ctype not in items:
                continue
            txt = (items[ctype].get("content") or "").strip()
            if not txt:
                continue
            metrics = compute_human_kgain_for_item(items[ctype].get("human_annotations") or [], canonical_qas)
            if metrics["kgain"] is None:
                continue
            scores[ctype] = metrics
            texts[ctype] = txt

        if len(scores) < 2:
            continue

        # Build RM prompt (same for all pairs for this article_id)
        prompt = build_rm_prompt(abstract_text, canonical_qas, include_correct=args.include_correct_in_prompt)

        # Build ordered candidate list by kgain
        cand = [(ctype, float(scores[ctype]["kgain"])) for ctype in scores.keys()]
        cand.sort(key=lambda x: x[1], reverse=True)

        def emit_pair(chosen_type: str, rejected_type: str):
            chosen_kg = float(scores[chosen_type]["kgain"])
            rejected_kg = float(scores[rejected_type]["kgain"])
            if chosen_kg - rejected_kg < args.min_margin:
                return
            rm_rows.append({
                "article_id": aid,
                "prompt": prompt,
                "chosen": texts[chosen_type],
                "rejected": texts[rejected_type],
                "meta": {
                    "chosen_type": chosen_type,
                    "rejected_type": rejected_type,
                    "chosen_human_kgain": chosen_kg,
                    "rejected_human_kgain": rejected_kg,
                    "all_scores": {k: float(v["kgain"]) for k, v in scores.items()},
                },
            })

        if args.pairs_mode == "best_vs_worst":
            emit_pair(cand[0][0], cand[-1][0])

        elif args.pairs_mode == "all_ordered":
            # every ordered pair with margin
            for i in range(len(cand)):
                for j in range(len(cand)):
                    if i == j:
                        continue
                    emit_pair(cand[i][0], cand[j][0])

        elif args.pairs_mode == "anchor_news":
            if "news" in scores:
                for other in scores.keys():
                    if other == "news":
                        continue
                    emit_pair("news", other)
            else:
                # fallback if news missing: best vs worst
                emit_pair(cand[0][0], cand[-1][0])

    dump_jsonl(rm_rows, args.rm_output)
    dump_jsonl(rl_rows, args.rl_output)

    print(f"Wrote {len(rm_rows)} RM preference pairs to {args.rm_output}")
    print(f"Wrote {len(rl_rows)} RL records to {args.rl_output}")


if __name__ == "__main__":
    main()
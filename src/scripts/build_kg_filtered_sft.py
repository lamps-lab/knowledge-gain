#!/usr/bin/env python3
import argparse
import json
import statistics


def qas_to_text(qas):
    blocks = []
    for qa in qas or []:
        qid = qa.get("question_in_set")
        q = qa.get("question_text") or qa.get("question-text") or ""
        opts = "\n".join([f"  {i+1}. {o}" for i, o in enumerate(qa.get("options", []))])
        blocks.append(f"Q{qid}: {q}\n{opts}")
    return "\n\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_kg", type=float, default=None)
    ap.add_argument("--top_percent", type=float, default=None, help="Example: 0.75 keeps top 75% by KG")
    ap.add_argument("--use_qas", action="store_true")
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if args.top_percent is not None:
        kgs = sorted([
            float(r["llmsim_kg"])
            for r in rows
            if r.get("llmsim_kg") is not None
        ], reverse=True)
        cutoff_idx = max(0, min(len(kgs) - 1, int(len(kgs) * args.top_percent) - 1))
        threshold = kgs[cutoff_idx]
    elif args.min_kg is not None:
        threshold = args.min_kg
    else:
        threshold = -999.0

    kept = []
    for r in rows:
        kg_raw = r.get("llmsim_kg", None)
        if kg_raw is None:
            continue
        try:
            kg = float(kg_raw)
        except Exception:
            continue
        if kg < threshold:
            continue

        abstract = r["paper_abstract"]
        article = r["best_article"]
        qas = r["qa_annotations"]

        system = (
            "You are an expert science journalist. "
            "Write a clear, accurate, engaging science news article. "
            "Output ONLY the article text."
        )

        if args.use_qas:
            user = (
                f"ABSTRACT:\n{abstract}\n\n"
                f"TARGET QUESTIONS THE ARTICLE SHOULD HELP READERS ANSWER:\n"
                f"{qas_to_text(qas)}\n\n"
                f"Write a science news article grounded strictly in the abstract."
            )
        else:
            user = (
                f"ABSTRACT:\n{abstract}\n\n"
                f"Write a science news article grounded strictly in the abstract."
            )

        kept.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": article},
            ],
            "article_id": r.get("article_id"),
            "llmsim_kg": kg,
            "llmsim_pre_acc": r.get("llmsim_pre_acc"),
            "llmsim_post_acc": r.get("llmsim_post_acc"),
        })

    with open(args.out, "w", encoding="utf-8") as w:
        for r in kept:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Input rows: {len(rows)}")
    print(f"KG threshold: {threshold:.4f}")
    print(f"Kept rows: {len(kept)}")
    if kept:
        print(f"Mean kept KG: {statistics.mean([x['llmsim_kg'] for x in kept]):.4f}")


if __name__ == "__main__":
    main()
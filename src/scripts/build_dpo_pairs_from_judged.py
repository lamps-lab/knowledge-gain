#!/usr/bin/env python3
import argparse
import json


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
    ap.add_argument("--use_qas", action="store_true")
    ap.add_argument("--min_kg", type=float, default=None)
    args = ap.parse_args()

    n = 0
    with open(args.input, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)

            if args.min_kg is not None and float(ex.get("llmsim_kg", 0.0)) < args.min_kg:
                continue

            candidates = ex.get("candidates", [])
            if not candidates:
                continue

            # Reject the lowest open-source judged candidate.
            rejected = sorted(candidates, key=lambda c: float(c.get("total", 0.0)))[0]["article"]
            chosen = ex["best_article"]

            if chosen.strip() == rejected.strip():
                continue

            system = (
                "You are an expert science journalist. "
                "Write a clear, accurate, engaging science news article. "
                "Output ONLY the article text."
            )

            if args.use_qas:
                user = (
                    f"ABSTRACT:\n{ex['paper_abstract']}\n\n"
                    f"TARGET QUESTIONS THE ARTICLE SHOULD HELP READERS ANSWER:\n"
                    f"{qas_to_text(ex['qa_annotations'])}\n\n"
                    f"Write a science news article grounded strictly in the abstract."
                )
            else:
                user = (
                    f"ABSTRACT:\n{ex['paper_abstract']}\n\n"
                    f"Write a science news article grounded strictly in the abstract."
                )

            rec = {
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "chosen": chosen,
                "rejected": rejected,
                "article_id": ex.get("article_id"),
                "llmsim_kg": ex.get("llmsim_kg"),
            }

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Saved {n} DPO pairs to {args.out}")


if __name__ == "__main__":
    main()
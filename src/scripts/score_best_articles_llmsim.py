#!/usr/bin/env python3
import argparse
import json
import os

from llmsim_reward_cached import LLMSimKGRewardCached


def count_existing_lines(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    json.loads(line)
                    n += 1
                except Exception:
                    # Ignore malformed partial final line, if any.
                    pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="judged_teacher_news.jsonl")
    ap.add_argument("--out", required=True, help="Output scored JSONL")
    ap.add_argument("--n_simulated_readers", type=int, default=2)
    ap.add_argument("--pre_cache", default="cache/pre_cache_train.jsonl")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    already_done = count_existing_lines(args.out) if args.resume else 0
    mode = "a" if args.resume else "w"

    print(f"Resume mode: {args.resume}")
    print(f"Existing completed output rows: {already_done}")
    print(f"Writing mode: {mode}")

    scorer = LLMSimKGRewardCached(
        n_simulated_readers=args.n_simulated_readers,
        seed=42,
        pre_cache_path=args.pre_cache,
    )

    done_this_run = 0
    total_seen = 0

    with open(args.input, "r", encoding="utf-8") as f, open(args.out, mode, encoding="utf-8") as w:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            # Skip rows already written in previous crashed run.
            if idx < already_done:
                continue

            if args.max_examples is not None and done_this_run >= args.max_examples:
                break

            total_seen += 1
            ex = json.loads(line)

            try:
                abstract = ex["paper_abstract"]
                qas = ex["qa_annotations"]
                article = ex["best_article"]

                score = scorer.score_article(
                    abstract=abstract,
                    article=article,
                    qas=qas,
                    return_details=False,
                )

                ex["llmsim_kg"] = score["kg"]
                ex["llmsim_pre_acc"] = score["pre_acc"]
                ex["llmsim_post_acc"] = score["post_acc"]
                ex["llmsim_n_readers"] = args.n_simulated_readers
                ex["llmsim_error"] = None

            except Exception as e:
                # Do not crash the whole run. Save the row with error metadata.
                ex["llmsim_kg"] = None
                ex["llmsim_pre_acc"] = None
                ex["llmsim_post_acc"] = None
                ex["llmsim_n_readers"] = args.n_simulated_readers
                ex["llmsim_error"] = repr(e)
                print(f"[ERROR] idx={idx} article_id={ex.get('article_id')} error={repr(e)}")

            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
            w.flush()

            done_this_run += 1

            print(
                f"[{idx + 1}] article_id={ex.get('article_id')} "
                f"KG={ex.get('llmsim_kg')} "
                f"pre={ex.get('llmsim_pre_acc')} "
                f"post={ex.get('llmsim_post_acc')}"
            )

    print(f"Done. Added {done_this_run} new rows.")
    print(f"Saved scored examples to {args.out}")


if __name__ == "__main__":
    main()
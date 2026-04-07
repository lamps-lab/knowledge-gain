#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_score", type=float, default=0.0)
    ap.add_argument("--min_accuracy", type=int, default=4)
    ap.add_argument("--min_completeness", type=int, default=3)
    ap.add_argument("--min_clarity", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    kept = 0
    seen = 0

    with open(args.teacher_records, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            row = json.loads(line)

            if not row.get("accepted_for_sft", False):
                continue

            scores = row.get("best_scores") or {}
            total = float(row.get("best_weighted_total", -999.0))

            if total < args.min_score:
                continue
            if int(scores.get("accuracy", 0)) < args.min_accuracy:
                continue
            if int(scores.get("completeness", 0)) < args.min_completeness:
                continue
            if int(scores.get("clarity", 0)) < args.min_clarity:
                continue

            out_row = {
                "article_id": row.get("article_id"),
                "paper_abstract": row["paper_abstract"],
                "target_article": row["best_article"],
                "teacher_source": row.get("best_source"),
                "teacher_scores": scores,
                "teacher_weighted_total": total,
            }
            w.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"kept {kept} / {seen}", flush=True)


if __name__ == "__main__":
    main()
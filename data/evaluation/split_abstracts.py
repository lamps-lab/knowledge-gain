#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # JSON array
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array at top level.")
        return data

    # JSONL
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any] | None:
    out = dict(rec)

    abstract = out.get("paper_abstract") or out.get("abstract") or ""
    abstract = normalize_ws(str(abstract))

    if not abstract:
        return None

    out["paper_abstract"] = abstract

    # Ensure article_id exists
    if "article_id" not in out or out["article_id"] in (None, ""):
        # fallback priority
        fallback = out.get("id") or out.get("news_url") or out.get("abstract_url")
        out["article_id"] = fallback

    return out


def dedupe_records(records: List[Dict[str, Any]], dedupe_by: str) -> List[Dict[str, Any]]:
    if dedupe_by == "none":
        return records

    seen = set()
    kept = []

    for rec in records:
        if dedupe_by == "article_id":
            key = str(rec.get("article_id", "")).strip()
        elif dedupe_by == "abstract":
            key = normalize_ws(rec["paper_abstract"]).lower()
        else:
            raise ValueError(f"Unknown dedupe mode: {dedupe_by}")

        if not key:
            kept.append(rec)
            continue

        if key in seen:
            continue
        seen.add(key)
        kept.append(rec)

    return kept


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(name: str, rows: List[Dict[str, Any]]) -> None:
    print(f"{name}: {len(rows)} records")
    cat_counts = Counter(str(r.get("category", "UNKNOWN")) for r in rows)
    top = cat_counts.most_common(10)
    if top:
        print(f"{name} top categories: {top}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSON array or JSONL file")
    ap.add_argument("--train_out", required=True, help="Output JSONL for train split")
    ap.add_argument("--val_out", required=True, help="Output JSONL for val split")
    ap.add_argument("--val_count", type=int, default=None, help="Exact number of validation examples")
    ap.add_argument("--val_fraction", type=float, default=0.10, help="Validation fraction if val_count is not set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--dedupe_by",
        choices=["none", "article_id", "abstract"],
        default="abstract",
        help="How to deduplicate before splitting",
    )
    args = ap.parse_args()

    raw = load_records(args.input)

    normalized = []
    dropped_empty = 0
    for rec in raw:
        nr = normalize_record(rec)
        if nr is None:
            dropped_empty += 1
            continue
        normalized.append(nr)

    deduped = dedupe_records(normalized, args.dedupe_by)

    rng = random.Random(args.seed)
    rng.shuffle(deduped)

    n = len(deduped)
    if n == 0:
        raise ValueError("No usable records found after normalization/filtering.")

    if args.val_count is not None:
        val_count = args.val_count
    else:
        val_count = max(1, int(round(n * args.val_fraction)))

    if val_count <= 0 or val_count >= n:
        raise ValueError(f"Invalid val size: {val_count} for dataset size {n}")

    val_rows = deduped[:val_count]
    train_rows = deduped[val_count:]

    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.val_out, val_rows)

    print("Done.")
    print(f"Input records: {len(raw)}")
    print(f"Dropped empty abstracts: {dropped_empty}")
    print(f"After normalization: {len(normalized)}")
    print(f"After dedupe ({args.dedupe_by}): {len(deduped)}")
    summarize("train", train_rows)
    summarize("val", val_rows)


if __name__ == "__main__":
    main()
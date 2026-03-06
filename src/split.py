#!/usr/bin/env python3
"""
split_80_20.py

Creates consistent 80/20 train/test splits for:
- rm_pairs.jsonl   (multiple lines per article_id)
- rl_records.jsonl (one line per article_id)

IMPORTANT: Splits by article_id so all pairs/records for a paper stay in the same split.

Usage:
  python split_80_20.py \
    --rm rm_pairs.jsonl \
    --rl rl_records.jsonl \
    --out_dir splits \
    --test_ratio 0.2 \
    --seed 0
"""

from __future__ import annotations
import argparse, json, os, random
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_aid(row: Dict[str, Any]) -> str:
    # support common keys
    aid = row.get("article_id", row.get("article-id", row.get("articleId", None)))
    if aid is None:
        return ""
    return str(aid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm", required=True, help="rm_pairs.jsonl")
    ap.add_argument("--rl", required=True, help="rl_records.jsonl")
    ap.add_argument("--out_dir", default="splits")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rm_rows = read_jsonl(args.rm)
    rl_rows = read_jsonl(args.rl)

    # Group RM pairs by article_id
    rm_by_aid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rm_rows:
        aid = get_aid(r)
        if not aid:
            continue
        rm_by_aid[aid].append(r)

    # RL records by article_id
    rl_by_aid: Dict[str, Dict[str, Any]] = {}
    for r in rl_rows:
        aid = get_aid(r)
        if not aid:
            continue
        rl_by_aid[aid] = r

    # Intersection of article_ids present in BOTH sets (recommended)
    aids = sorted(set(rm_by_aid.keys()) & set(rl_by_aid.keys()))
    if not aids:
        raise SystemExit("No overlapping article_ids found between RM and RL files.")

    rng = random.Random(args.seed)
    rng.shuffle(aids)

    n_test = max(1, int(round(len(aids) * args.test_ratio)))
    test_aids = set(aids[:n_test])
    train_aids = set(aids[n_test:])

    # Build splits
    rm_train, rm_test = [], []
    for aid in aids:
        if aid in test_aids:
            rm_test.extend(rm_by_aid[aid])
        else:
            rm_train.extend(rm_by_aid[aid])

    rl_train = [rl_by_aid[aid] for aid in aids if aid in train_aids]
    rl_test  = [rl_by_aid[aid] for aid in aids if aid in test_aids]

    # Write
    rm_train_path = os.path.join(args.out_dir, "rm_pairs_train.jsonl")
    rm_test_path  = os.path.join(args.out_dir, "rm_pairs_test.jsonl")
    rl_train_path = os.path.join(args.out_dir, "rl_records_train.jsonl")
    rl_test_path  = os.path.join(args.out_dir, "rl_records_test.jsonl")

    write_jsonl(rm_train, rm_train_path)
    write_jsonl(rm_test, rm_test_path)
    write_jsonl(rl_train, rl_train_path)
    write_jsonl(rl_test, rl_test_path)

    print("Done.")
    print(f"Article IDs total: {len(aids)} | train: {len(train_aids)} | test: {len(test_aids)}")
    print(f"RM pairs  total: {len(rm_rows)} | train: {len(rm_train)} | test: {len(rm_test)}")
    print(f"RL recs   total: {len(rl_rows)} | train: {len(rl_train)} | test: {len(rl_test)}")
    print("Wrote:")
    print(" ", rm_train_path)
    print(" ", rm_test_path)
    print(" ", rl_train_path)
    print(" ", rl_test_path)


if __name__ == "__main__":
    main()

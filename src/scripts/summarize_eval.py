#!/usr/bin/env python3
import argparse
import json
import statistics


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(name, rows):
    kgs = [float(r["kg"]) for r in rows]
    pres = [float(r["pre_acc"]) for r in rows]
    posts = [float(r["post_acc"]) for r in rows]

    print(f"\n{name}")
    print(f"N = {len(rows)}")
    print(f"Mean KG = {statistics.mean(kgs):.4f}")
    print(f"Median KG = {statistics.median(kgs):.4f}")
    print(f"Mean PRE = {statistics.mean(pres):.4f}")
    print(f"Mean POST = {statistics.mean(posts):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()

    for path in args.files:
        summarize(path, load(path))


if __name__ == "__main__":
    main()
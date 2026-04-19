#!/usr/bin/env python3
import argparse
import json
import statistics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("llmsim_kg") is not None:
                rows.append(r)

    kgs = [float(r["llmsim_kg"]) for r in rows]
    pres = [float(r["llmsim_pre_acc"]) for r in rows]
    posts = [float(r["llmsim_post_acc"]) for r in rows]

    print(f"N = {len(rows)}")
    print(f"Mean KG = {statistics.mean(kgs):.4f}")
    print(f"Median KG = {statistics.median(kgs):.4f}")
    print(f"Mean PRE = {statistics.mean(pres):.4f}")
    print(f"Mean POST = {statistics.mean(posts):.4f}")
    print(f"% positive KG = {sum(x > 0 for x in kgs) / len(kgs):.2%}")
    print(f"% zero KG = {sum(x == 0 for x in kgs) / len(kgs):.2%}")
    print(f"% negative KG = {sum(x < 0 for x in kgs) / len(kgs):.2%}")

if __name__ == "__main__":
    main()
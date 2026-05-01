#!/usr/bin/env python3
# scripts/compare_eval_linewise.py

import argparse
import json
import random
import statistics
from math import sqrt

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bootstrap_ci(diffs, n_boot=10000, seed=42):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="SFT-All eval jsonl", default="results/eval_sft_all_s2_s5.jsonl")
    ap.add_argument("--new", help="KG-filtered eval jsonl", default="results/eval_sft_kg_filtered_top75_s5.jsonl")
    args = ap.parse_args()

    base = load(args.base)
    new = load(args.new)

    n = min(len(base), len(new))
    base = base[:n]
    new = new[:n]

    base_kgs = [float(r["kg"]) for r in base]
    new_kgs = [float(r["kg"]) for r in new]
    diffs = [nkg - bkg for bkg, nkg in zip(base_kgs, new_kgs)]

    print(f"N paired = {n}")
    print(f"Base mean KG = {statistics.mean(base_kgs):.4f}")
    print(f"New mean KG = {statistics.mean(new_kgs):.4f}")
    print(f"Mean diff = {statistics.mean(diffs):.4f}")
    print(f"Median diff = {statistics.median(diffs):.4f}")

    ci_lo, ci_hi = bootstrap_ci(diffs)
    print(f"Bootstrap 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

    sd = statistics.stdev(diffs)
    se = sd / sqrt(n)
    print(f"Paired diff SD = {sd:.4f}")
    print(f"Paired diff SE = {se:.4f}")

    print(f"New > Base count = {sum(d > 0 for d in diffs)}")
    print(f"New = Base count = {sum(d == 0 for d in diffs)}")
    print(f"New < Base count = {sum(d < 0 for d in diffs)}")

    if wilcoxon is not None:
        stat, p_greater = wilcoxon(diffs, alternative="greater")
        stat2, p_two = wilcoxon(diffs, alternative="two-sided")
        print(f"Wilcoxon one-sided p(new > base) = {p_greater:.6g}")
        print(f"Wilcoxon two-sided p = {p_two:.6g}")


if __name__ == "__main__":
    main()
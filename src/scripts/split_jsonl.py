#!/usr/bin/env python3
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--test_size", type=int, default=300)
    ap.add_argument("--dev_size", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    random.Random(args.seed).shuffle(rows)

    test = rows[:args.test_size]
    dev = rows[args.test_size:args.test_size + args.dev_size]
    train = rows[args.test_size + args.dev_size:]

    for path, subset in [(args.train, train), (args.dev, dev), (args.test, test)]:
        with open(path, "w", encoding="utf-8") as w:
            for r in subset:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path}: {len(subset)}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import json
from collections import Counter


INPUT_JSON = "collected_abstracts.json"


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = []
    for row in data:
        category = str(row.get("category", "")).strip()
        if not category:
            category = "MISSING"
        categories.append(category)

    counts = Counter(categories)
    total = sum(counts.values())

    print(f"Total abstracts: {total}\n")
    print("Category distribution:")
    for category, count in counts.most_common():
        pct = (count / total * 100) if total else 0
        print(f"{category}: {count} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
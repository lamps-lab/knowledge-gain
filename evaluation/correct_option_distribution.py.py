import json
from pathlib import Path
from collections import Counter, defaultdict


INPUT_PATH = Path("kgain_questions_with_news0_news2.json")
OUTPUT_PATH = Path("correct_option_distribution.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    data = load_json(INPUT_PATH)

    overall_counter = Counter()
    by_article = {}
    by_category = defaultdict(Counter)

    total_questions = 0

    for item in data:
        article_id = item["article_id"]
        category = item.get("category", "UNKNOWN")
        qa_annotations = item.get("qa_annotations", [])

        article_counter = Counter()

        for q in qa_annotations:
            correct_option = q.get("correct_option")

            if correct_option is None:
                continue

            correct_option = int(correct_option)

            overall_counter[correct_option] += 1
            article_counter[correct_option] += 1
            by_category[category][correct_option] += 1
            total_questions += 1

        by_article[str(article_id)] = {
            "category": category,
            "num_questions": sum(article_counter.values()),
            "correct_option_counts": dict(sorted(article_counter.items())),
        }

    overall_distribution = {
        option: {
            "count": count,
            "percentage": round((count / total_questions) * 100, 2)
        }
        for option, count in sorted(overall_counter.items())
    }

    category_distribution = {}

    for category, counter in sorted(by_category.items()):
        category_total = sum(counter.values())
        category_distribution[category] = {
            option: {
                "count": count,
                "percentage": round((count / category_total) * 100, 2)
            }
            for option, count in sorted(counter.items())
        }

    output = {
        "total_articles": len(data),
        "total_questions": total_questions,
        "overall_correct_option_distribution": overall_distribution,
        "by_category": category_distribution,
        "by_article": by_article,
    }

    save_json(output, OUTPUT_PATH)

    print(f"Total articles: {len(data)}")
    print(f"Total questions: {total_questions}")
    print("\nOverall correct option distribution:")

    for option, stats in overall_distribution.items():
        print(
            f"Option {option}: "
            f"{stats['count']} questions "
            f"({stats['percentage']}%)"
        )

    print(f"\nSaved distribution to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
import json
from pathlib import Path


GENERATED_QUESTIONS_PATH = Path("generated_questions.json")
EVAL_DATASET_PATH = Path("eval_dataset.json")
OUTPUT_PATH = Path("kgain_questions_with_news0_news2.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    generated_questions = load_json(GENERATED_QUESTIONS_PATH)
    eval_dataset = load_json(EVAL_DATASET_PATH)

    # If either file is stored as {"data": [...]}, unwrap it.
    if isinstance(generated_questions, dict):
        generated_questions = generated_questions.get("data", generated_questions)

    if isinstance(eval_dataset, dict):
        eval_dataset = eval_dataset.get("data", eval_dataset)

    # Build lookup from eval_dataset by article/triplet id.
    eval_by_id = {
        int(item["id"]): item
        for item in eval_dataset
    }

    merged = []

    for q_item in generated_questions:
        article_id = int(q_item["article_id"])

        if article_id not in eval_by_id:
            print(f"Warning: article_id {article_id} not found in eval_dataset.json")
            continue

        eval_item = eval_by_id[article_id]

        merged_item = {
            # Main identifiers
            "article_id": article_id,
            "date": eval_item.get("date"),
            "category": eval_item.get("category"),

            # Source metadata
            "news_url": eval_item.get("news_url"),
            "abstract_url": eval_item.get("abstract_url"),

            # Abstract / paper text
            "paper_abstract": (
                q_item.get("paper_abstract")
                or eval_item.get("abstract")
            ),

            # Generated knowledge questions
            "qa_annotations": q_item.get("qa_annotations", []),

            # Keep only desired news versions
            "news_0": eval_item.get("news_0"),
            "news_2": eval_item.get("news_2"),
        }

        merged.append(merged_item)

    # Sort by article_id
    merged = sorted(merged, key=lambda x: x["article_id"])

    save_json(merged, OUTPUT_PATH)

    print(f"Saved {len(merged)} merged records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
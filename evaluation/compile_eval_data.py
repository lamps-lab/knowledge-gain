import json

FILE_0 = "./sft_outputs/generated_news_70.json"
FILE_1 = "simple_news_4b.json"
FILE_2 = "agentic_qwen_4b.json"
OUTPUT_FILE = "eval_dataset_top50.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_valid_field(field_name, *records):
    for rec in records:
        val = rec.get(field_name)
        if val:  # Evaluates to False if None or empty string ""
            return val
    return None


def main():
    data_0 = load_json(FILE_0)
    data_1 = load_json(FILE_1)
    data_2 = load_json(FILE_2)

    by_id_0 = {x["id"]: x for x in data_0}
    by_id_1 = {x["id"]: x for x in data_1}
    by_id_2 = {x["id"]: x for x in data_2}

    all_ids = sorted(set(by_id_0) | set(by_id_1) | set(by_id_2))
    merged = []

    for article_id in all_ids:
        rec0 = by_id_0.get(article_id, {})
        rec1 = by_id_1.get(article_id, {})
        rec2 = by_id_2.get(article_id, {})

        date = get_valid_field("date", rec0, rec1, rec2)
        category = get_valid_field("category", rec0, rec1, rec2)
        news_url = get_valid_field("news_url", rec0, rec1, rec2)
        abstract_url = get_valid_field("abstract_url", rec0, rec1, rec2)
        
        
        abstract = get_valid_field("abstract", rec0, rec1, rec2) or get_valid_field("paper_abstract", rec0, rec1, rec2)

        news_3 = get_valid_field("news", rec0, rec1, rec2) or get_valid_field("news_article", rec0, rec1, rec2) or ""

        news_0 = get_valid_field("news_0", rec0) or ""
        news_1 = get_valid_field("news_1", rec1) or ""
        news_2 = get_valid_field("news_2", rec2) or ""

        merged.append({
            "id": article_id,
            "date": date,
            "category": category,
            "news_url": news_url,
            "abstract": abstract,
            "abstract_url": abstract_url,
            "news_0": news_0.strip(),
            "news_1": news_1.strip(),
            "news_2": news_2.strip(),
            "news_3": news_3.strip(),
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(merged)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

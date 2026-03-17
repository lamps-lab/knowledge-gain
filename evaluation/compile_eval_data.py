import json

FILE_0 = "generated_news.json"
FILE_1 = "generated_simple.json"
FILE_2 = "agentic_news.json"
OUTPUT_FILE = "eval_dataset.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_base(*records):
    for rec in records:
        if rec:
            return rec
    return {}


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

        base = pick_base(rec0, rec1, rec2)
        if not base:
            continue

        merged.append({
            "id": base.get("id"),
            "date": base.get("date"),
            "category": base.get("category"),
            "news_url": base.get("news_url"),
            "abstract": base.get("abstract"),
            "abstract_url": base.get("abstract_url"),
            "news_0": (rec0.get("news_0") or "").strip(),
            "news_1": (rec1.get("news_1") or "").strip(),
            "news_2": (rec2.get("news_2") or "").strip(),
            "news_3": (base.get("news") or "").strip(),
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(merged)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
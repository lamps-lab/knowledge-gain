#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List

from openai import OpenAI


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    txt = open(path, "r", encoding="utf-8").read().strip()
    if not txt:
        return []
    if "\n" in txt and txt.lstrip().startswith("{") and txt.rstrip().endswith("}"):
        try:
            obj = json.loads(txt)
            return [obj] if isinstance(obj, dict) else list(obj)
        except Exception:
            return [json.loads(line) for line in txt.splitlines() if line.strip()]
    obj = json.loads(txt)
    return [obj] if isinstance(obj, dict) else list(obj)


def generate_article(client: OpenAI, model: str, abstract: str, news_length: int) -> str:
    system = (
        "You are an expert science journalist. "
        "Write a clear, engaging science news article for a general audience. "
        "Output only the article text."
    )

    user = (
        f"Write a science news article of about {news_length} words based on this abstract.\n\n"
        f"{abstract}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
    )

    return (resp.choices[0].message.content or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../evaluation/twenty_samples.json")
    ap.add_argument("--output", default="generated_simple.json")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    recs = read_json_or_jsonl(args.input)
    if not recs:
        raise SystemExit("No records found.")

    client = OpenAI()
    outs = []

    for i, rec in enumerate(recs):
        article_id = rec.get("id", 0)
        category = rec.get("category", "")
        abstract = (rec.get("abstract") or "").strip()
        news = (rec.get("news") or "").strip()

        if not abstract or "Manual extraction needed" in abstract:
            print(f"Skipping {i+1}/{len(recs)} | ID: {article_id} | no valid abstract")
            outs.append(rec)
            continue

        news_length = len(news.split())

        print(f"Processing {i+1}/{len(recs)} | ID: {article_id} | {category} | target={news_length} words")

        generated_news = generate_article(client, args.model, abstract, news_length)

        rec["generated_news"] = generated_news

        outs.append(rec)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(outs, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
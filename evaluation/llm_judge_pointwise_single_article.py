#!/usr/bin/env python3
import json
import random
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

INPUT_FILE = "eval_dataset.json"
OUTPUT_FILE = "llm_judge_single_article.json"
MODEL = "gpt-5-mini"
SEED = 42
MAX_TOKENS = 1200
MAX_RETRIES = 1

ARTICLE_KEYS = ["news_0", "news_1", "news_2", "news_3"]

ARTICLE_SCHEMA = {
    "type": "object",
    "properties": {
        "accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
        "accuracy_reason": {"type": "string"},
        "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
        "completeness_reason": {"type": "string"},
        "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
        "relevance_reason": {"type": "string"},
        "clarity": {"type": "integer", "minimum": 1, "maximum": 5},
        "clarity_reason": {"type": "string"},
    },
    "required": [
        "accuracy",
        "accuracy_reason",
        "completeness",
        "completeness_reason",
        "relevance",
        "relevance_reason",
        "clarity",
        "clarity_reason",
    ],
    "additionalProperties": False,
}


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_score(d: Dict[str, Any]) -> float:
    vals = [d["accuracy"], d["completeness"], d["relevance"], d["clarity"]]
    return round(sum(vals) / 4.0, 3)


def chat_create_json(
    client: OpenAI,
    model: str,
    system_text: str,
    user_text: str,
    schema_name: str,
    schema: Dict[str, Any],
    max_tokens: int = MAX_TOKENS,
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                #temperature=0,
                #max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            )

            out = (resp.choices[0].message.content or "").strip()
            if not out:
                raise RuntimeError("Empty message.content")
            return json.loads(out)

        except Exception as e:
            last_err = e
            time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_err}")


def build_prompt(abstract: str, article: str) -> str:
    return f"""You are evaluating a science news article for a general audience.

Evaluate the article independently using the abstract as the reference for scientific content.
Use the 1-5 anchored scales below.

Accuracy
1 = Major factual errors, invented claims, or serious contradictions of the abstract
2 = Noticeable inaccuracies, overclaims, or unsupported details
3 = Mostly accurate, but with some meaningful imprecision or unsupported framing
4 = Accurate overall, with only minor issues
5 = Fully accurate and faithful to the abstract, with no material errors

Completeness
1 = Misses most of the main findings or scientific ideas
2 = Covers some important points but leaves out major findings, mechanisms, or caveats
3 = Covers the core result, but misses several important details
4 = Covers the main findings and most key scientific points
5 = Thoroughly covers the key findings, mechanisms, caveats, and important context from the abstract

Relevance
1 = Focuses on trivial, off-topic, or low-value details; fails to show why the work matters
2 = Somewhat relevant, but misses the main significance or news angle
3 = Moderately relevant; identifies the main result but only partly explains why it matters
4 = Clearly emphasizes the important and newsworthy aspects
5 = Strongly foregrounds the most significant and newsworthy aspects of the work

Clarity
1 = Very difficult to follow for a general audience
2 = Often unclear, jargon-heavy, or poorly structured
3 = Generally understandable, but uneven or sometimes confusing
4 = Clear and easy to follow, with good explanation of technical ideas
5 = Exceptionally clear, readable, and accessible without losing scientific meaning

Guidelines:
- Judge only this one article, not any hypothetical alternatives.
- Keep reasons brief and specific.
- Do not reward style alone if the science is wrong.
- Do not penalize the article for omitting information absent from the abstract.
- Base Accuracy and Completeness only on the abstract.
- For Relevance and Clarity, judge the article as science communication for a general audience.

ABSTRACT:
{abstract}

ARTICLE:
{article}
"""


def judge_one_article(
    client: OpenAI,
    abstract: str,
    article: str,
) -> Dict[str, Any]:
    system = (
        "You are a careful evaluator of science communication. "
        "Be strict about factual faithfulness and scientific coverage, "
        "but fair about stylistic variation."
    )

    user = build_prompt(abstract, article)

    result = chat_create_json(
        client=client,
        model=MODEL,
        system_text=system,
        user_text=user,
        schema_name="single_article_pointwise_eval_v1",
        schema=ARTICLE_SCHEMA,
    )
    result["mean_score"] = mean_score(result)
    return result


def judge_record(client: OpenAI, rec: Dict[str, Any]) -> Dict[str, Any]:
    article_order = ARTICLE_KEYS[:]
    rng = random.Random(SEED + int(rec["id"]))
    rng.shuffle(article_order)

    scores_by_system = {}

    for key in article_order:
        article = (rec.get(key) or "").strip()
        if not article:
            continue

        scores_by_system[key] = judge_one_article(
            client=client,
            abstract=rec["abstract"],
            article=article,
        )

    return {
        "id": rec.get("id"),
        "date": rec.get("date"),
        "category": rec.get("category"),
        "news_url": rec.get("news_url"),
        "abstract_url": rec.get("abstract_url"),
        "judging_order": article_order,
        "scores_by_system": scores_by_system,
    }


def main():
    dataset = load_json(INPUT_FILE)
    client = OpenAI()
    outputs = []

    for i, rec in enumerate(dataset):
        print(f"Judging {i+1}/{len(dataset)} | ID={rec.get('id')} | {rec.get('category')}")
        result = judge_record(client, rec)
        outputs.append(result)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved {len(outputs)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
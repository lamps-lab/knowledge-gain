#!/usr/bin/env python3
import json
import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

from openai import OpenAI

INPUT_FILE = "eval_dataset.json"
OUTPUT_FILE = "llm_judge_pointwise.json"
MODEL = "gpt-5-mini"
SEED = 42
MAX_OUTPUT_TOKENS = 4000
MAX_RETRIES = 1

ARTICLE_KEYS = ["news_0", "news_1", "news_2", "news_3"]


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


ARTICLE_RATING_SCHEMA = {
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

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "article_A": deepcopy(ARTICLE_RATING_SCHEMA),
        "article_B": deepcopy(ARTICLE_RATING_SCHEMA),
        "article_C": deepcopy(ARTICLE_RATING_SCHEMA),
        "article_D": deepcopy(ARTICLE_RATING_SCHEMA),
    },
    "required": ["article_A", "article_B", "article_C", "article_D"],
    "additionalProperties": False,
}


def responses_create_json(
    client: OpenAI,
    model: str,
    system_text: str,
    user_text: str,
    schema_name: str,
    schema: Dict[str, Any],
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
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
                #max_tokens=max_output_tokens,
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

def build_label_map(article_id: int) -> Dict[str, str]:
    keys = ARTICLE_KEYS[:]
    rng = random.Random(SEED + int(article_id))
    rng.shuffle(keys)
    return {
        "A": keys[0],
        "B": keys[1],
        "C": keys[2],
        "D": keys[3],
    }


def build_prompt(abstract: str, articles_by_label: Dict[str, str]) -> str:
    return f"""You are evaluating science news articles for a general audience.

The four articles below describe the same underlying paper. Their order has been randomized and blinded.
Evaluate each article independently. Do not rank them. Ties are allowed.
Use the abstract as the reference for scientific content.

Rate each article on a 1-5 scale for these dimensions:

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

Keep each reason brief and specific.
Do not let writing style alone inflate Accuracy or Completeness.
Do not penalize an article for omitting information that is absent from the abstract.

ABSTRACT:
{abstract}

ARTICLE A:
{articles_by_label["A"]}

ARTICLE B:
{articles_by_label["B"]}

ARTICLE C:
{articles_by_label["C"]}

ARTICLE D:
{articles_by_label["D"]}
"""


def mean_score(d: Dict[str, Any]) -> float:
    vals = [d["accuracy"], d["completeness"], d["relevance"], d["clarity"]]
    return round(sum(vals) / 4.0, 3)


def judge_record(client: OpenAI, rec: Dict[str, Any]) -> Dict[str, Any]:
    for k in ARTICLE_KEYS:
        if not rec.get(k):
            raise ValueError(f"Record id={rec.get('id')} missing {k}")

    label_map = build_label_map(int(rec["id"]))
    articles_by_label = {
        label: rec[source_key].strip()
        for label, source_key in label_map.items()
    }

    system = (
        "You are a careful evaluator of science communication. "
        "Judge each article independently against the abstract. "
        "Be strict about factual faithfulness, but fair about stylistic variation."
    )

    user = build_prompt(rec["abstract"], articles_by_label)

    raw = responses_create_json(
        client=client,
        model=MODEL,
        system_text=system,
        user_text=user,
        schema_name="pointwise_news_eval_v1",
        schema=JUDGE_SCHEMA,
    )

    scores_by_label = {
        "A": raw["article_A"],
        "B": raw["article_B"],
        "C": raw["article_C"],
        "D": raw["article_D"],
    }

    scores_by_system = {}
    for label, system_key in label_map.items():
        item = scores_by_label[label]
        scores_by_system[system_key] = {
            **item,
            "mean_score": mean_score(item),
        }

    return {
        "id": rec.get("id"),
        "date": rec.get("date"),
        "category": rec.get("category"),
        "news_url": rec.get("news_url"),
        "abstract_url": rec.get("abstract_url"),
        "label_map": label_map,
        "scores_by_label": scores_by_label,
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
#!/usr/bin/env python3

import json
import random
import time
from typing import Any, Dict, List, Optional

import anthropic

INPUT_FILE = "eval_dataset.json"
OUTPUT_FILE = "llm_judge_pairwise_news1_claude_sonnet.json"
MODEL = "claude-sonnet-4-6"
SEED = 42
MAX_TOKENS = 1000
MAX_RETRIES = 1

ARTICLE_KEYS = ["news_0", "news_1", "news_2", "news_3"]
BASELINE_KEY = "news_1"
COMPARE_KEYS = [k for k in ARTICLE_KEYS if k != BASELINE_KEY]

PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {
        "better_article": {
            "type": "string",
            "enum": ["article_a", "article_b", "tie"],
            "description": "Which article is better overall."
        },
        "reason": {
            "type": "string",
            "description": "Brief comparative explanation grounded in the abstract and the two articles."
        }
    },
    "required": ["better_article", "reason"],
    "additionalProperties": False,
}


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_from_message(resp) -> str:
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def claude_create_json(
    client: anthropic.Anthropic,
    model: str,
    system_text: str,
    user_text: str,
    schema: Dict[str, Any],
    max_tokens: int = MAX_TOKENS,
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                system=system_text,
                max_tokens=max_tokens,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": user_text,
                    }
                ],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                },
            )

            if getattr(resp, "stop_reason", None) == "max_tokens":
                raise RuntimeError("Claude hit max_tokens before finishing structured output")

            out = extract_text_from_message(resp)
            if not out:
                raise RuntimeError("Empty text content")

            return json.loads(out)

        except Exception as e:
            last_err = e
            time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_err}")


def build_pairwise_prompt(abstract: str, article_a: str, article_b: str) -> str:
    return f"""
Compare the two science news articles using the abstract as the reference for scientific content.

Your task:
- Decide which article is better overall for a general audience.
- "Better overall" should reflect the total quality of the article as science communication:
  faithfulness to the abstract, coverage of key findings/context, relevance/newsworthiness,
  and clarity/readability.
- Do NOT score separate dimensions.
- Do NOT average hidden sub-scores.
- Prefer the article that is more scientifically faithful if one article is better written
  but less accurate.
- Use only the information provided in the abstract and the two articles.
- Do not rely on outside knowledge.
- The article order is randomized; do not let position affect your judgment.
- If the two articles are genuinely too close to call, return "tie".
- Keep the written reason brief, specific, and comparative.

<input>
<abstract><![CDATA[
{abstract}
]]></abstract>

<article_a><![CDATA[
{article_a}
]]></article_a>

<article_b><![CDATA[
{article_b}
]]></article_b>
</input>

Respond ONLY with a valid JSON object in exactly this format:
{{
  "better_article": "article_a" | "article_b" | "tie",
  "reason": "<1-3 sentences of comparative reasoning>"
}}
""".strip()


def judge_pair(
    client: anthropic.Anthropic,
    abstract: str,
    article_a: str,
    article_b: str,
) -> Dict[str, Any]:
    system = (
        "You are an expert science communicator and journal editor comparing two science "
        "news articles and deciding which one better represents the original abstract "
        "for a general audience."
    )

    user = build_pairwise_prompt(abstract, article_a, article_b)

    return claude_create_json(
        client=client,
        model=MODEL,
        system_text=system,
        user_text=user,
        schema=PAIRWISE_SCHEMA,
    )


def map_winner_to_key(
    better_article: str,
    article_a_key: str,
    article_b_key: str,
) -> Optional[str]:
    if better_article == "article_a":
        return article_a_key
    if better_article == "article_b":
        return article_b_key
    return None


def judge_record(client: anthropic.Anthropic, rec: Dict[str, Any]) -> Dict[str, Any]:
    baseline_article = (rec.get(BASELINE_KEY) or "").strip()

    comparisons = []
    baseline_wins = 0
    baseline_losses = 0
    baseline_ties = 0
    skipped_pairs = []

    if not baseline_article:
        return {
            "id": rec.get("id"),
            "date": rec.get("date"),
            "category": rec.get("category"),
            "news_url": rec.get("news_url"),
            "abstract_url": rec.get("abstract_url"),
            "baseline_key": BASELINE_KEY,
            "comparisons": [],
            "summary": {
                "baseline_wins": 0,
                "baseline_losses": 0,
                "baseline_ties": 0,
                "skipped_pairs": COMPARE_KEYS[:],
                "note": f"Baseline article '{BASELINE_KEY}' was empty or missing."
            },
        }

    for other_key in COMPARE_KEYS:
        other_article = (rec.get(other_key) or "").strip()
        if not other_article:
            skipped_pairs.append(other_key)
            continue

        presented_keys = [BASELINE_KEY, other_key]
        rng = random.Random(f"{SEED}:{rec.get('id')}:{other_key}")
        rng.shuffle(presented_keys)

        article_a_key, article_b_key = presented_keys
        article_a_text = (rec.get(article_a_key) or "").strip()
        article_b_text = (rec.get(article_b_key) or "").strip()

        judgment = judge_pair(
            client=client,
            abstract=rec["abstract"],
            article_a=article_a_text,
            article_b=article_b_text,
        )

        winner_key = map_winner_to_key(
            judgment["better_article"],
            article_a_key=article_a_key,
            article_b_key=article_b_key,
        )

        if winner_key == BASELINE_KEY:
            baseline_wins += 1
            baseline_result = "win"
        elif winner_key is None:
            baseline_ties += 1
            baseline_result = "tie"
        else:
            baseline_losses += 1
            baseline_result = "loss"

        comparisons.append(
            {
                "pair": [BASELINE_KEY, other_key],
                "presented_order": {
                    "article_a": article_a_key,
                    "article_b": article_b_key,
                },
                "winner_label": judgment["better_article"],
                "winner_key": winner_key,
                "baseline_result": baseline_result,
                "judgment": judgment,
            }
        )

    return {
        "id": rec.get("id"),
        "date": rec.get("date"),
        "category": rec.get("category"),
        "news_url": rec.get("news_url"),
        "abstract_url": rec.get("abstract_url"),
        "baseline_key": BASELINE_KEY,
        "comparisons": comparisons,
        "summary": {
            "baseline_wins": baseline_wins,
            "baseline_losses": baseline_losses,
            "baseline_ties": baseline_ties,
            "skipped_pairs": skipped_pairs,
        },
    }


def main():
    dataset = load_json(INPUT_FILE)
    client = anthropic.Anthropic()
    outputs = []

    for i, rec in enumerate(dataset):
        print(
            f"Judging {i+1}/{len(dataset)} | "
            f"ID={rec.get('id')} | {rec.get('category')} | baseline={BASELINE_KEY}"
        )

        result = judge_record(client, rec)
        outputs.append(result)

        # checkpoint save
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(f"Saved {len(outputs)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
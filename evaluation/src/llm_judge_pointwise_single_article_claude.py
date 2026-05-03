#!/usr/bin/env python3
import json
import random
import time
from typing import Any, Dict, List, Optional

import anthropic

INPUT_FILE = f"../eval_dataset_top50.json"
GENERATED_QUESTIONS_FILE = "../generated_questions.json"
OUTPUT_FILE = f"../results/open-source/pointwise_top50.json"

MODEL = "claude-sonnet-4-6"
SEED = 42
MAX_TOKENS = 1800
MAX_RETRIES = 1

ARTICLE_KEYS = ["news_0", "news_1", "news_2", "news_3"]

LIKERT_INT = {
    "type": "integer",
    "enum": [1, 2, 3, 4, 5],
    "description": "Likert score from 1 to 5 inclusive.",
}

ARTICLE_SCHEMA = {
    "type": "object",
    "properties": {
        "accuracy": LIKERT_INT,
        "accuracy_reason": {"type": "string"},
        "completeness": LIKERT_INT,
        "completeness_reason": {"type": "string"},
        "relevance": LIKERT_INT,
        "relevance_reason": {"type": "string"},
        "clarity": LIKERT_INT,
        "clarity_reason": {"type": "string"},
        "knowledge_gain": LIKERT_INT,
        "knowledge_gain_reason": {"type": "string"},
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
        "knowledge_gain",
        "knowledge_gain_reason",
    ],
    "additionalProperties": False,
}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_score(d: Dict[str, Any]) -> float:
    vals = [
        d["accuracy"],
        d["completeness"],
        d["relevance"],
        d["clarity"],
        d["knowledge_gain"],
    ]
    return round(sum(vals) / 5.0, 3)


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


def get_record_id(rec: Dict[str, Any]) -> Optional[int]:
    val = rec.get("id", rec.get("article_id"))
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def index_generated_questions(
    generated_questions: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}

    for item in generated_questions:
        article_id = item.get("article_id")
        qa_annotations = item.get("qa_annotations", [])

        try:
            article_id = int(article_id)
        except (TypeError, ValueError):
            continue

        if not isinstance(qa_annotations, list):
            qa_annotations = []

        out[article_id] = qa_annotations

    return out


def format_questions_for_prompt(qa_annotations: List[Dict[str, Any]]) -> str:
    if not qa_annotations:
        return "No generated questions were provided for this article."

    chunks = []
    for i, qa in enumerate(qa_annotations, start=1):
        q_num = qa.get("question_in_set", i)
        question_text = (qa.get("question-text") or "").strip()
        options = qa.get("options", [])
        correct_answer = qa.get("correct_answer")

        lines = [f'<question index="{q_num}">']
        lines.append(f"<stem>{question_text}</stem>")

        if isinstance(options, list) and options:
            lines.append("<options>")
            for opt in options:
                lines.append(f"- {str(opt).strip()}")
            lines.append("</options>")

        if correct_answer is not None:
            lines.append(f"<correct_answer>{str(correct_answer).strip()}</correct_answer>")

        lines.append("</question>")
        chunks.append("\n".join(lines))

    return "\n\n".join(chunks)


def build_prompt(abstract: str, article: str, qa_annotations: List[Dict[str, Any]]) -> str:
    questions_block = format_questions_for_prompt(qa_annotations)

    return f"""Evaluate the article independently using the abstract as the reference for scientific content, while also estimating knowledge gain using the generated question set.

Instructions:
- Evaluate each dimension independently.
- Base Accuracy and Completeness only on the abstract.
- Base Relevance and Clarity on how well the article communicates the abstract’s actual findings to a general audience.
- Base Knowledge Gain only on what a high-school-level reader could answer from the article itself, using the generated questions as the probe.
- Do not use the abstract to rescue missing article content in Knowledge Gain.
- Do not reward unsupported embellishment, fabricated quotes, invented mechanisms, or added real-world implications.
- Penalize hallucinations in Accuracy first, then reduce Relevance, Clarity, and Knowledge Gain.
- If the article contains hallucinations or invented details, apply a penalty to Accuracy first, and also reduce Relevance, Clarity, and Knowledge Gain when the hallucination affects trust or understanding.
- Minor paraphrase is acceptable only if the meaning stays fully grounded in the abstract.
- If uncertain between two adjacent scores, prefer the lower score unless the evidence for the higher score is explicit in the article.
- Unsupported but plausible additions count as hallucinations if they are not stated in the abstract.

<rubric>
  <dimension name="Accuracy">
    <anchor score="5">Fully faithful to the abstract. No invented claims, no fabricated quotes, no unsupported mechanisms, and no meaningful overinterpretation.</anchor>
    <anchor score="4">Mostly faithful, with only minor paraphrase or very light interpretation that does not change the scientific meaning.</anchor>
    <anchor score="3">Main claim is correct, but the article adds at least one unsupported detail, overstatement, or interpretation that weakens faithfulness.</anchor>
    <anchor score="2">Contains multiple unsupported additions or one serious distortion of the abstract’s meaning.</anchor>
    <anchor score="1">Major factual errors, fabricated details, or topic drift away from the abstract.</anchor>
  </dimension>

  <dimension name="Completeness">
    <anchor score="1">Misses most of the main findings or scientific ideas.</anchor>
    <anchor score="2">Covers some important points but leaves out major findings, mechanisms, caveats, or essential context present in the abstract.</anchor>
    <anchor score="3">Covers the core result, but omits several important supporting details, mechanisms, caveats, or context needed for a fuller understanding.</anchor>
    <anchor score="4">Covers the main findings and most key scientific points from the abstract.</anchor>
    <anchor score="5">Thoroughly covers the key findings, mechanisms, caveats, and important context from the abstract.</anchor>
  </dimension>

  <dimension name="Relevance">
    <anchor score="1">Focuses on trivial, off-topic, or low-value details and fails to show why the work matters.</anchor>
    <anchor score="2">Somewhat relevant, but misses the main significance, implication, or news angle.</anchor>
    <anchor score="3">Identifies the main result but only partly explains why it matters to a general audience.</anchor>
    <anchor score="4">Clearly emphasizes the important and newsworthy aspects of the work.</anchor>
    <anchor score="5">Strongly foregrounds the most significant and newsworthy aspects of the work for a general audience.</anchor>
  </dimension>

  <dimension name="Clarity">
    <anchor score="1">Very difficult to follow for a general audience.</anchor>
    <anchor score="2">Often unclear, jargon-heavy, poorly explained, or poorly structured.</anchor>
    <anchor score="3">Generally understandable, but uneven, occasionally confusing, or insufficiently explanatory for non-experts.</anchor>
    <anchor score="4">Clear and easy to follow, with good explanation of technical ideas.</anchor>
    <anchor score="5">Exceptionally clear, readable, and accessible without losing scientific meaning.</anchor>
  </dimension>

  <dimension name="Knowledge Gain">
    <anchor score="5">The article explicitly states nearly all facts needed to answer the question set. A reader could answer most questions correctly using only the article.</anchor>
    <anchor score="4">The article states most facts needed for the question set, with only a few missing or ambiguous points.</anchor>
    <anchor score="3">The article covers some key facts, but several questions would still be hard to answer from the article alone.</anchor>
    <anchor score="2">The article provides only a small subset of the facts needed for the question set.</anchor>
    <anchor score="1">The article provides almost none of the facts needed for the question set.</anchor>
  </dimension>
</rubric>

<input>
  <abstract><![CDATA[
{abstract}
  ]]></abstract>

  <article><![CDATA[
{article}
  ]]></article>

  <generated_questions><![CDATA[
{questions_block}
  ]]></generated_questions>
</input>

Respond ONLY with a valid JSON object in exactly this format:
{{
  "accuracy_reason": "<1-2 sentences of evidence-based reasoning>",
  "accuracy": <1-5>,
  "completeness_reason": "<1-2 sentences of evidence-based reasoning>",
  "completeness": <1-5>,
  "relevance_reason": "<1-2 sentences of evidence-based reasoning>",
  "relevance": <1-5>,
  "clarity_reason": "<1-2 sentences of evidence-based reasoning>",
  "clarity": <1-5>,
  "knowledge_gain_reason": "<1-2 sentences of evidence-based reasoning>"
  "knowledge_gain": <1-5>,
}}
"""


def judge_one_article(
    client: anthropic.Anthropic,
    abstract: str,
    article: str,
    qa_annotations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    system = (
        "You are an expert science communicator and journal editor evaluating whether "
        "a science news article faithfully and accessibly represents the findings in "
        "the original abstract, and how much knowledge a general reader would gain."
    )

    user = build_prompt(
        abstract=abstract,
        article=article,
        qa_annotations=qa_annotations,
    )

    result = claude_create_json(
        client=client,
        model=MODEL,
        system_text=system,
        user_text=user,
        schema=ARTICLE_SCHEMA,
    )
    result["mean_score"] = mean_score(result)
    return result


def judge_record(
    client: anthropic.Anthropic,
    rec: Dict[str, Any],
    questions_index: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    article_order = ARTICLE_KEYS[:]

    rec_id = get_record_id(rec)
    seed_id = rec_id if rec_id is not None else 0

    rng = random.Random(SEED + seed_id)
    rng.shuffle(article_order)

    qa_annotations = questions_index.get(seed_id, [])
    scores_by_system = {}

    for key in article_order:
        article = (rec.get(key) or "").strip()
        if not article:
            continue

        scores_by_system[key] = judge_one_article(
            client=client,
            abstract=(rec.get("abstract") or "").strip(),
            article=article,
            qa_annotations=qa_annotations,
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
    generated_questions = load_json(GENERATED_QUESTIONS_FILE)
    questions_index = index_generated_questions(generated_questions)

    client = anthropic.Anthropic()
    outputs = []

    for i, rec in enumerate(dataset):
        rec_id = rec.get("id", rec.get("article_id"))
        print(f"Judging {i+1}/{len(dataset)} | ID={rec_id} | {rec.get('category')}")

        result = judge_record(
            client=client,
            rec=rec,
            questions_index=questions_index,
        )
        outputs.append(result)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(f"Saved {len(outputs)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

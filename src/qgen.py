#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMP_GEN = float(os.getenv("TEMP_GEN", "0.7"))
TEMP_VERIFY = float(os.getenv("TEMP_VERIFY", "0.2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
MAX_ROUNDS = int(os.getenv("QGEN_MAX_ROUNDS", "2"))  # generation + up to 2 verify/repair rounds
IDK_TEXT = "I do not know the answer."

# few-shot file path
FEWSHOT_PATH = os.getenv("QGEN_FEWSHOT_PATH", "../data/qgen/fewshot.json").strip()

QGEN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "article_id": {"type": "integer"},
        "qa_annotations": {
            "type": "array",
            "minItems": 6,
            "maxItems": 9,
            "items": {
                "type": "object",
                "properties": {
                    "question_in_set": {"type": "integer", "minimum": 1},
                    "question-text": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
                    "correct_option": {"type": "integer", "minimum": 1, "maximum": 4},
                    "correct_answer": {"type": "string"},
                },
                "required": ["question_in_set", "question-text", "options", "correct_option", "correct_answer"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["article_id", "qa_annotations"],
    "additionalProperties": False,
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_ok": {"type": "boolean"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_in_set": {"type": "integer"},
                    "ok": {"type": "boolean"},
                    "problems": {"type": "array", "items": {"type": "string"}},
                    "replacement": {
                        "type": ["object", "null"],
                        "properties": {
                            "question-text": {"type": "string"},
                            "options": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
                            "correct_option": {"type": "integer", "minimum": 1, "maximum": 4},
                            "correct_answer": {"type": "string"},
                        },
                        "required": ["question-text", "options", "correct_option", "correct_answer"],
                        "additionalProperties": False,
                    },
                },
                "required": ["question_in_set", "ok", "problems", "replacement"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["overall_ok", "items", "notes"],
    "additionalProperties": False,
}

# Prompts
SYSTEM_QGEN = f"""You are a multiple-choice question generator for a human study.

Your job:
- Read BOTH the paper abstract and the news article.
- Generate a question set that tests GENERAL KNOWLEDGE about the finding/concepts.
- Questions must be SELF-CONTAINED (no references to sources).

Hard constraints:
1) SELF-CONTAINED:
   Do NOT use phrases like: "the study", "the paper", "the abstract", "the article",
   "the researchers", "according to", "this research".
2) OVERLAPPING KNOWLEDGE:
   Each question must be answerable with the SAME correct answer by someone who read ONLY the abstract
   OR ONLY the news article. (Overlap only; not source-specific.)
3) NO STUDY/RESEARCHER DETAILS:
   Avoid authors, venue, affiliations, study design minutiae, apparatus specs that aren't a general claim.
4) FORMAT:
   - First 2 True/False questions (TF)
   - Next 2 Easy MCQ questions (Easy)
   - Next 2 Hard MCQ questions (Hard)
   (Total = 6 questions.)
5) OPTIONS:
   - TF options must be exactly: ["True","False","{IDK_TEXT}"]
     correct_option must be 1 or 2.
   - MCQ options must be exactly 5 options, with last option exactly "{IDK_TEXT}".
     correct_option must be 1..4.
6) Easy vs Hard:
   - Easy: directly stated in BOTH texts.
   - Hard: not a verbatim span, but still inferable from BOTH texts.

Return STRICT JSON in the required schema, nothing else.
"""

USER_QGEN_TEMPLATE = """PAPER ABSTRACT:
{paper_abstract}

NEWS ARTICLE:
{news_article}

Generate the 6 questions now (2 TF, 2 Easy MCQ, 2 Hard MCQ) in that order.
Use question_in_set = 1..6 in order.
Return JSON only.
"""

SYSTEM_VERIFY = f"""You are a strict verifier and repairer for the generated question set.

Check EACH question against these rules:

A) Self-contained:
   Must NOT reference sources ("the study/paper/article/abstract/researchers/according to", etc.)
B) Overlap:
   Must be answerable from ONLY the abstract AND also from ONLY the news article
   (same correct answer; no source-specific details).
C) Format:
   - Q1–Q2 are TF: options exactly ["True","False","{IDK_TEXT}"]; correct_option in {{1,2}}
   - Q3–Q4 are Easy MCQ: 5 options; last is "{IDK_TEXT}"; correct_option in {{1..4}}
   - Q5–Q6 are Hard MCQ: 5 options; last is "{IDK_TEXT}"; correct_option in {{1..4}}
D) No researcher/venue/study-protocol minutiae.

If a question is invalid, produce ONE replacement with the SAME question_in_set that satisfies all rules.
Keep difficulty level consistent with its slot (TF/Easy/Hard).
Return JSON only using the verification schema.
"""

USER_VERIFY_TEMPLATE = """PAPER ABSTRACT:
{paper_abstract}

NEWS ARTICLE:
{news_article}

DRAFT QUESTIONS JSON:
{draft_json}

Verify each question and provide replacements for invalid ones.
Return JSON only.
"""

def responses_create_json(
    client: OpenAI,
    system_text: str,
    user_text: str,
    schema_name: str,
    schema: Dict[str, Any],
    temperature: float,
    max_output_tokens: int = 1800,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            #print("RUNNING FILE:", __file__)
            #print("SCHEMA NAME:", schema_name)
            #print("VERIFY REQUIRED:", schema.get("properties", {}).get("items", {}).get("items", {}).get("required"))
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            out = (resp.output_text or "").strip()
            if not out:
                raise RuntimeError("Empty output_text.")
            return json.loads(out)
        except Exception as e:
            last_err = e
            time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_err}")

def load_fewshot_block() -> str:
    if not FEWSHOT_PATH:
        return ""
    try:
        obj = json.load(open(FEWSHOT_PATH, "r", encoding="utf-8"))
        ex_abs = obj.get("paper_abstract", "")
        ex_news = obj.get("news_article", "")
        ex_out = obj.get("good_output", {})
        if not ex_abs or not ex_news or not ex_out:
            return ""
        return (
            "\n\n=== GOOD EXAMPLE (few-shot) ===\n"
            "PAPER ABSTRACT:\n" + ex_abs + "\n\n"
            "NEWS ARTICLE:\n" + ex_news + "\n\n"
            "GOOD OUTPUT JSON:\n" + json.dumps(ex_out, ensure_ascii=False, indent=2) + "\n"
            "=== END EXAMPLE ===\n"
        )
    except Exception:
        return ""

def extract_texts(record: Dict[str, Any]) -> Tuple[int, str, str]:
    """
    Supports common patterns:
      - {"article_id": 31, "paper_abstract": "...", "news_article": "..."}
      - {"article_id": 31, "abstract": "...", "news": "..."}
      - {"article_id": 31, "abstract_item": {"content": "..."}, "news_item": {"content": "..."}}
    """
    article_id = record.get("article_id", record.get("id", 0))
    if not isinstance(article_id, int):
        try:
            article_id = int(article_id)
        except Exception:
            article_id = 0

    abs_txt = record.get("paper_abstract") or record.get("abstract") or ""
    news_txt = record.get("news_article") or record.get("news") or ""

    if not abs_txt and isinstance(record.get("abstract_item"), dict):
        abs_txt = record["abstract_item"].get("content", "") or ""
    if not news_txt and isinstance(record.get("news_item"), dict):
        news_txt = record["news_item"].get("content", "") or ""

    return article_id, str(abs_txt).strip(), str(news_txt).strip()

def apply_repairs(draft: Dict[str, Any], ver: Dict[str, Any]) -> Dict[str, Any]:
    idx_by_q = {it["question_in_set"]: i for i, it in enumerate(draft["qa_annotations"])}

    for item in ver.get("items", []):
        if item.get("ok", True):
            continue
        qn = item["question_in_set"]
        rep = item.get("replacement")
        if not rep:
            continue
        if qn not in idx_by_q:
            continue
        i = idx_by_q[qn]
        draft["qa_annotations"][i]["question-text"] = rep["question-text"]
        draft["qa_annotations"][i]["options"] = rep["options"]
        draft["qa_annotations"][i]["correct_option"] = rep["correct_option"]
        draft["qa_annotations"][i]["correct_answer"] = rep["correct_answer"]

    return draft

def qgen(client: OpenAI, article_id: int, paper_abstract: str, news_article: str) -> Dict[str, Any]:
    fewshot = load_fewshot_block()

    # 1) generate
    draft = responses_create_json(
        client=client,
        system_text=SYSTEM_QGEN + fewshot,
        user_text=USER_QGEN_TEMPLATE.format(paper_abstract=paper_abstract, news_article=news_article),
        schema_name="qgen",
        schema=QGEN_SCHEMA,
        temperature=TEMP_GEN,
        max_output_tokens=1800,
    )
    draft["article_id"] = article_id  # trust input id

    # 2) verify/repair (1–2 rounds)
    for _ in range(MAX_ROUNDS):
        ver = responses_create_json(
            client=client,
            system_text=SYSTEM_VERIFY,
            user_text=USER_VERIFY_TEMPLATE.format(
                paper_abstract=paper_abstract,
                news_article=news_article,
                draft_json=json.dumps(draft, ensure_ascii=False, indent=2),
            ),
            schema_name="verify",
            schema=VERIFY_SCHEMA,
            temperature=TEMP_VERIFY,
            max_output_tokens=1800,
        )
        if ver.get("overall_ok", False):
            break
        draft = apply_repairs(draft, ver)

    draft["paper_abstract"] = paper_abstract
    draft["news_article"] = news_article
    return draft

def load_records(path: str) -> List[Dict[str, Any]]:
    txt = open(path, "r", encoding="utf-8").read().strip()
    if not txt:
        return []
    # JSONL if multiple lines that look like json objects
    if "\n" in txt and txt.lstrip().startswith("{") and txt.rstrip().endswith("}"):
        # try json first
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return [obj]
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        # else treat as jsonl
        recs = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
        return recs
    # fallback: standard json
    obj = json.loads(txt)
    if isinstance(obj, dict):
        return [obj]
    return list(obj)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON or JSONL with article_id + abstract + news")
    ap.add_argument("--output", default="", help="Output path (jsonl). If empty, print JSON.")
    args = ap.parse_args()

    recs = load_records(args.input)
    if not recs:
        raise SystemExit("No records found in input.")

    client = OpenAI()

    outs: List[Dict[str, Any]] = []
    for rec in recs:
        article_id, abs_txt, news_txt = extract_texts(rec)
        if not abs_txt or not news_txt:
            raise SystemExit(
                "Record missing abstract/news. Provide keys like paper_abstract + news_article "
                "or abstract + news or abstract_item/news_item with content."
            )
        out = qgen(client, article_id, abs_txt, news_txt)
        outs.append(out)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as w:
            json.dump(outs if len(outs) > 1 else outs[0], w, ensure_ascii=False, indent=2)
            w.write("\n")
    else:
        if len(outs) == 1:
            print(json.dumps(outs[0], ensure_ascii=False, indent=2))
        else:
            print(json.dumps(outs, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
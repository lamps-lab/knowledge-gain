#!/usr/bin/env python3
import argparse, json, time
from typing import Any, Dict, List, Optional

from openai import OpenAI

CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {"type": "array", "minItems": 1, "items": {"type": "string"}}
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

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

def responses_create_json(
    client: OpenAI,
    model: str,
    system_text: str,
    user_text: str,
    schema_name: str,
    schema: Dict[str, Any],
    temperature: float,
    max_output_tokens: int = 1800,
    max_retries: int = 5,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role": "system", "content": system_text},
                       {"role": "user", "content": user_text}],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                text={"format": {"type": "json_schema",
                                 "name": schema_name,
                                 "schema": schema,
                                 "strict": True}},
            )
            out = (resp.output_text or "").strip()
            if not out:
                raise RuntimeError("Empty output_text")
            return json.loads(out)
        except Exception as e:
            last_err = e
            time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    raise RuntimeError(f"Failed after {max_retries} attempts: {last_err}")

def gen_candidates(client: OpenAI, model: str, abstract: str, k: int, temp: float) -> List[str]:
    MIN_WORDS = 450
    MAX_WORDS = 750
    PARAGRAPH_GUIDE = "12–20 short paragraphs (often 1–3 sentences each), separated by blank lines"
    HEADLINE_GUIDE = "Start with a punchy headline on the first line (no period)."

    system = (
        "You are a science news writer.\n"
        "Write a clear, engaging news article for a general audience.\n"
        "It is OK to mention that findings come from a study/paper.\n"
        "Do NOT mention 'the abstract' or that you were given an abstract.\n"
        "Do NOT invent facts: no made-up numbers, cohorts, years, institutions, author names, or journal names.\n"
        "Only include a journal name if it is explicitly present in the abstract text.\n"
        "Prefer short paragraphs and a journalistic tone.\n"
    )

    user = (
        f"ABSTRACT:\n{abstract}\n\n"
        f"Generate {k} distinct news-article drafts matching this style:\n"
        f"- {HEADLINE_GUIDE}\n"
        f"- Length: {MIN_WORDS}–{MAX_WORDS} words\n"
        f"- Structure: {PARAGRAPH_GUIDE}\n"
        f"- Tone: explanatory, confident-but-not-hype, with concrete takeaways\n\n"
        "Content requirements:\n"
        "- Lead with the main finding in plain language.\n"
        "- Include 1–3 brief context paragraphs explaining why it matters.\n"
        "- Include key quantitative results ONLY if present in the abstract; otherwise stay qualitative.\n"
        "- Avoid deep protocol minutiae (no optimizer names, no measurement internals, etc.).\n"
        "- If mentioning publication, keep it generic unless the abstract names the journal.\n\n"
        "Return JSON only: {\"candidates\":[...]}."
    )

    obj = responses_create_json(
        client=client,
        model=model,
        system_text=system,
        user_text=user,
        schema_name="news_candidates_v2",  # keep a new name to avoid schema caching weirdness
        schema=CANDIDATES_SCHEMA,
        temperature=temp,
        max_output_tokens=2200,
    )
    cands = [c.strip() for c in obj["candidates"] if isinstance(c, str) and c.strip()]
    return cands[:k] if len(cands) > k else cands

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out.json", help="out.json / qgen output JSON/JSONL with paper_abstract, qa_annotations, and optionally news_article")
    ap.add_argument("--output", default="candidates.json", help="JSONL to write: one record per abstract with candidates")
    ap.add_argument("--k", type=int, default=6, help="Generated candidates per abstract")
    ap.add_argument("--include_original", action="store_true", help="Also include original news_article as a candidate")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temp", type=float, default=0.9)
    args = ap.parse_args()

    recs = read_json_or_jsonl(args.input)
    if not recs:
        raise SystemExit("No records found.")

    client = OpenAI()
    outs = []

    with open(args.output, "w", encoding="utf-8") as w:
        for rec in recs:
            abstract = rec.get("paper_abstract") or rec.get("abstract") or ""
            qas = rec.get("qa_annotations") or []
            if not abstract or not qas:
                continue

            candidates = []
            if args.include_original and rec.get("news_article"):
                candidates.append({"source": "original", "text": rec["news_article"]})

            gen = gen_candidates(client, args.model, abstract, args.k, args.temp)
            for i, t in enumerate(gen):
                candidates.append({"source": f"gen_{i}", "text": t})

            outs.append({
                "article_id": rec.get("article_id", rec.get("id", 0)),
                "paper_abstract": abstract,
                "qa_annotations": qas,
                "candidates": candidates,
            })
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(outs, f, ensure_ascii=False, indent=2)
        f.write("\n")

if __name__ == "__main__":
    main()
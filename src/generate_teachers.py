#!/usr/bin/env python3
import argparse, json, requests, re

import re

def sanitize_article(text):
    """Strips out internal monologue, headers, and metadata from the article text."""
    # 1. Remove anything before the first "Title:" or "Headline:" or the actual start of a paragraph
    # If the model uses a clear marker like "Final Article:", we jump straight there
    if "Final Article:" in text:
        text = text.split("Final Article:")[-1]
    
    # 2. Strip out common reasoning headers
    garbage_patterns = [
        r"(?i)Thinking Process:.*?(?=\n\n|\*\*|\Z)", 
        r"(?i)Word Count Check:.*?(?=\n\n|\Z)",
        r"(?i)Drafting:.*?(?=\n\n|\Z)",
        r"(?i)Para \d:.*?(?=\n\n|\Z)",
        r"(?i)Total: ~?\d+ words.*",
        r"(?i)\*Revised Draft.*?\*"
    ]
    
    for pattern in garbage_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    
    # 3. Clean up leading/trailing whitespace and double line breaks
    return text.strip()

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"

def system_prompt():
    return """You are a professional science journalist. 
You provide responses in JSON format. 
To ensure quality, you must use the 'model_notes' field for your internal planning and word-count math. 
The 'article' field must contain ONLY the final polished news story, starting with a Headline.

Example Output:
{
  "model_notes": "Planning: Lead with vaccine equity, target 450 words...",
  "article_id": 123,
  "article": "HEADLINE: THE GLOBAL VACCINE RACE\\n\\nDespite global efforts, vaccine distribution remains...",
}"""


def user_prompt(ex):
    return f"""Based on the ABSTRACT below, write a 450-word news article for a general audience. 
Return your response as a JSON object with the keys: "article_id", "article", and "qa_annotations".

ABSTRACT:
{ex["paper_abstract"]}

ARTICLE_ID:
{ex["article_id"]}

JSON Output:"""

def call_llm(system, user, temperature=0.1):
    schema = {
        "type": "object",
        "properties": {
            "model_notes": {"type": "string"},
            "article_id": {"type": "integer"},
            "article": {"type": "string"},
            "qa_annotations": {"type": "array"}
        },
        "required": ["model_notes", "article_id", "article", "qa_annotations"]
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": 2000,
        "extra_body": {"guided_json": schema}
    }

    r = requests.post(URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ----------------------------
# JSON EXTRACTOR
# ----------------------------
def extract_article(text):
    try:
        obj = json.loads(text)
        if "article" in obj:
            return obj["article"]
    except:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return obj.get("article", "")
        except:
            pass

    return text.strip()


# ----------------------------
# MAIN
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data) if l.strip()]
    system = system_prompt()

    with open(args.out, "a") as w:
        for idx, ex in enumerate(records):

            raw = call_llm(system, user_prompt(ex), temperature=0.1)
            
            article = extract_article(raw)

            # retry if bad output
            if not article or len(article) < 200:
                raw = call_llm(system, user_prompt(ex), temperature=0.1)
                article = extract_article(raw)

            # ----------------------------
            # BUILD FINAL OUTPUT
            # ----------------------------
            ex["candidates"] = [{
                "source": "simple",
                "article": article
            }]

            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
            w.flush()

            print(f"[{idx+1}/{len(records)}] done")


if __name__ == "__main__":
    main()
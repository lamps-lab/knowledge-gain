#!/usr/bin/env python3
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

client = OpenAI()

def call_openai(system, user, model_name, temp):
    """Wrapper for OpenAI Chat Completion."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temp,
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return None

def process_record(ex, news_length, model_name):
    abstract = ex["paper_abstract"]
    qas = ex.get("qa_annotations", [])
    q_text = qas_to_text(qas)
    candidates = []

    # 2 Simple Samples
    simple_sys = "You are an expert science journalist. Write a clear news article. Output ONLY the article text starting with a headline. No meta-talk."
    simple_user = f"ABSTRACT:\n{abstract}\n\nWrite a {news_length} word article."
    
    for i in range(2):
        art = call_openai(simple_sys, simple_user, model_name, temp=0.2)
        if art:
            candidates.append({"source": "simple", "article": art})

    # 2 Agentic Samples (Draft -> Revise)
    drafter_sys = "You are a science journalist. Draft an engaging news article based on the abstract. Output ONLY the draft."
    drafter_user = f"ABSTRACT:\n{abstract}\n\nDraft a {news_length} word article."
    
    editor_sys = "You are a senior science editor. Revise the provided draft for factual accuracy and hook. Output ONLY the final polished article text. No labels like 'Revised Version:'."

    for i in range(2):
        draft = call_openai(drafter_sys, drafter_user, model_name, temp=0.5)
        if draft:
            editor_user = f"ORIGINAL ABSTRACT:\n{abstract}\n\nINITIAL DRAFT:\n{draft}\n\nRevise and output the final polished article."
            final_art = call_openai(editor_sys, editor_user, model_name, temp=0.2)
            if final_art:
                candidates.append({"source": "agentic", "article": final_art})
    
    ex["candidates"] = candidates
    return ex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--news_length", type=int, default=450)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    with open(args.data, 'r') as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"Generating 4 candidates (2 simple, 2 agentic) for {len(records)} records...")

    with open(args.out, "a", encoding="utf-8") as w:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_record, rec, args.news_length, args.model) for rec in records]
            
            for i, future in enumerate(futures):
                res = future.result()
                w.write(json.dumps(res, ensure_ascii=False) + "\n")
                w.flush()
                if (i + 1) % 5 == 0:
                    print(f"[{i+1}/{len(records)}] 4 candidates saved.")

if __name__ == "__main__":
    main()
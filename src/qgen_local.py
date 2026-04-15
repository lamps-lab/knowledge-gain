#!/usr/bin/env python3
import argparse
import json
import torch
import os
from typing import Dict
from transformers import AutoTokenizer
from tqdm import tqdm
from gptqmodel import GPTQModel

# CONFIG & PROMPTS
IDK_TEXT = "I do not know the answer."
MAX_ROUNDS = 1  

SYSTEM_QGEN = f"""You are a multiple-choice question generator for a human study.

Your job:
- Read the paper abstract.
- Generate a question set that tests GENERAL KNOWLEDGE about the findings/concepts.
- Questions must be phrased as UNIVERSAL TRIVIA, completely detached from the abstract itself.

Hard constraints:
1) SELF-CONTAINED (CRITICAL):
    Do NOT make it a reading comprehension test. 
    - FORBIDDEN NOUNS: "the study", "the paper", "the abstract", "the authors", "the data", "the dataset", "the model".
    - FORBIDDEN VERBS: "assessed", "reported", "stated", "observed", "measured", "estimated", "cited".
2) OVERLAPPING KNOWLEDGE:
    Each question must be answerable with the SAME correct answer by someone who read ONLY the abstract.
3) FORMAT:
    - Q1-Q2: True/False (TF). Options: ["True", "False", "{IDK_TEXT}"]. correct_option: 1 or 2.
    - Q3-Q4: Easy MCQ. 5 options; last is "{IDK_TEXT}". correct_option: 1-4.
    - Q5-Q6: Hard MCQ. 5 options; last is "{IDK_TEXT}". correct_option: 1-4.
(Total = 6 questions.)

Return STRICT JSON with keys: "article_id" (int), "qa_annotations" (list of objects).
Each object must have: "question_in_set", "question-text", "options", "correct_option", "correct_answer".
"""

SYSTEM_VERIFY = f"""You are a strict verifier and repairer for the generated question set.
Check EACH question against the "Self-contained" and "Universal Fact" rules. 
If ANY rule is violated (e.g. mentions "the study", "the dataset", or "the authors"), provide a replacement.

Format: Return JSON with "overall_ok" (bool) and "items" (list).
Each item needs: "question_in_set", "ok" (bool), "problems" (list), "replacement" (object or null).
"""

# MODEL UTILITIES

def load_local_model(model_name, device):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GPTQModel.from_quantized(model_name, device=device, dtype=torch.bfloat16)
    return model, tokenizer

def call_qwen_json(model, tokenizer, system_prompt, user_prompt) -> Dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1500, do_sample=False)
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        return json.loads(response[start:end])
    except Exception as e:
        return {}

# CORE LOGIC
def apply_repairs(draft: Dict, ver: Dict) -> Dict:
    idx_map = {it["question_in_set"]: i for i, it in enumerate(draft.get("qa_annotations", []))}
    for item in ver.get("items", []):
        if not item.get("ok", True) and item.get("replacement"):
            q_idx = item["question_in_set"]
            if q_idx in idx_map:
                draft["qa_annotations"][idx_map[q_idx]].update(item["replacement"])
    return draft

def process_single_abstract(model, tokenizer, article_id, abstract):
    user_prompt = f"PAPER ABSTRACT:\n{abstract}\n\nGenerate 6 questions (2 TF, 2 Easy, 2 Hard). Return JSON."
    draft = call_qwen_json(model, tokenizer, SYSTEM_QGEN, user_prompt)
    if not draft or "qa_annotations" not in draft:
        return None
    
    draft["article_id"] = article_id

    for _ in range(MAX_ROUNDS):
        verify_prompt = f"ABSTRACT:\n{abstract}\n\nDRAFT JSON:\n{json.dumps(draft)}\n\nVerify and repair if needed."
        ver_results = call_qwen_json(model, tokenizer, SYSTEM_VERIFY, verify_prompt)
        
        if ver_results.get("overall_ok"):
            break
        draft = apply_repairs(draft, ver_results)
        
    return draft

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            return json.loads(content)
        return [json.loads(l) for l in content.splitlines() if l.strip()]

def get_processed_abstracts(output_path) -> set:
    """Reads the existing output file and returns a set of completed abstract texts to allow resuming safely."""
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    row = json.loads(line)
                    if "paper_abstract" in row:
                        processed.add(row["paper_abstract"].strip())
                except Exception:
                    pass
    return processed

# MAIN LOOP
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4")
    args = parser.parse_args()

    model, tokenizer = load_local_model(args.model, "cuda:0")
    recs = load_data(args.input)

    # NEW RESUME LOGIC: Track by abstract text, not ID
    processed_abstracts = get_processed_abstracts(args.output)
    if processed_abstracts:
        print(f"Found {len(processed_abstracts)} already processed abstracts. Resuming...")

    with open(args.output, "a", encoding="utf-8") as w:
        for rec in tqdm(recs):
            
            # Use original ID
            article_id = rec.get("id", rec.get("article_id", 0))
            abstract = rec.get("abstract") or rec.get("paper_abstract")
            
            if not abstract:
                continue

            # Check if this exact abstract was already processed
            if abstract.strip() in processed_abstracts:
                continue

            result = process_single_abstract(model, tokenizer, article_id, abstract)
            
            if result:
                # Fix the underscore bug ("question_text" -> "question-text")
                for qa in result.get("qa_annotations", []):
                    if "question_text" in qa:
                        qa["question-text"] = qa.pop("question_text")

                result["paper_abstract"] = abstract 
                w.write(json.dumps(result, ensure_ascii=False) + "\n")
                w.flush()

if __name__ == "__main__":
    main()
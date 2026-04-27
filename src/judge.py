#!/usr/bin/env python3
import argparse, json, re, torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer

SCORE_RE = re.compile(r"A\s*=\s*([1-5]).*?COMP\s*=\s*([1-5]).*?REL\s*=\s*([1-5]).*?CLA\s*=\s*([1-5]).*?KG\s*=\s*([1-5])", re.I | re.S)

def qas_to_text(qas):
    """Formats the questions for the judge's prompt."""
    blocks = []
    for qa in qas or []:
        qid = qa.get("question_in_set")
        q = qa.get("question_text") or qa.get("question-text") or ""
        # We provide the correct answer so the judge knows what to look for in the article
        ans = qa.get("correct_answer") 
        blocks.append(f"Q{qid}: {q}\nCorrect Answer: {ans}")
    return "\n\n".join(blocks)

def parse_scores(text):
    m = SCORE_RE.search(text)
    if not m: return None
    try:
        a, c, r, cl, k = map(int, m.groups())
        return {"accuracy": a, "completeness": c, "relevance": r, "clarity": cl, "knowledge_gain": k}
    except:
        return None

def weighted_total(s, article):
    if not s: return -9.0
    # Weights optimized for Knowledge Gain (KG) and Accuracy (A)
    # Scale: 1=-1.0, 2=-0.5, 3=0, 4=0.5, 5=1.0
    a = (s['accuracy'] - 3) / 2.0
    k = (s['knowledge_gain'] - 3) / 2.0
    c = (s['completeness'] - 3) / 2.0
    cl = (s['clarity'] - 3) / 2.0
    r = (s['relevance'] - 3) / 2.0
    score = (0.30 * a) + (0.40 * k) + (0.10 * c) + (0.10 * cl) + (0.10 * r)
    if "Dr." in article:
        score -= 0.10
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 1}, attn_implementation="sdpa"
    )
    first_output_shown = False # Flag to show the first output

    with open(args.input) as f, open(args.out, "a") as w:
        for line in f:
            ex = json.loads(line)
            qa_text = qas_to_text(ex.get("qa_annotations", []))
            
            scored_candidates = []
            for cand in ex["candidates"]:
                sys_msg = (
                    "Task: Score this science article. Be extremely brief.\n"
                    "1. Accuracy (A): Matches abstract? (1=No, 5=Yes)\n"
                    "2. Completeness (COMP): Covers key points? (1=No, 5=Yes)\n"
                    "3. Relevance (REL): Good journalism? (1=No, 5=Yes)\n"
                    "4. Clarity (CLA): Clear writing? (1=No, 5=Yes)\n"
                    "5. Knowledge Gain (KG): Can reader answer all target questions using the article? (1=No, 5=Yes)\n\n"
                    "Output ONLY: A=X; COMP=X; REL=X; CLA=X; KG=X"
                )
                
                user_content = (
                    f"ABSTRACT:\n{ex['paper_abstract']}\n\n"
                    f"TARGET KNOWLEDGE (Questions/Answers):\n{qa_text}\n\n"
                    f"ARTICLE TO EVALUATE:\n{cand['article']}"
                )

                prompt = tokenizer.apply_chat_template([
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_content}
                ], tokenize=False, add_generation_prompt=True) + "A="

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
                    
                raw_out = "A=" + tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                if not first_output_shown:
                    print("\n" + "="*50)
                    print("DEBUG: FIRST RAW JUDGE RESPONSE")
                    print("="*50)
                    print(raw_out)
                    print("="*50 + "\n")
                    first_output_shown = True
                scores = parse_scores(raw_out)
                
                cand.update({"scores": scores, "total": weighted_total(scores,cand['article'])})
                scored_candidates.append(cand)
            
            scored_candidates.sort(key=lambda x: x["total"], reverse=True)
            ex["best_article"] = scored_candidates[0]["article"]
            ex["best_scores"] = scored_candidates[0]["scores"]
            
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
            w.flush()
            print(f"ID {ex.get('article_id')} | Best: {scored_candidates[0]['source']} | KG Score: {scored_candidates[0]['scores']['knowledge_gain']} | Acc: {scored_candidates[0]['scores']['accuracy']} | Rel: {scored_candidates[0]['scores']['relevance']}")

if __name__ == "__main__": main()
#!/usr/bin/env python3
import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def render_chat_prompt(tokenizer, system, user):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], 
        tokenize=False, 
        add_generation_prompt=True
    )

def generate_one(model, tokenizer, sys, usr, max_tokens, temp):
    prompt = render_chat_prompt(tokenizer, sys, usr)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **enc, 
            do_sample=temp > 0, 
            temperature=max(temp, 1e-5), 
            max_new_tokens=max_tokens, 
            use_cache=True, 
            pad_token_id=tokenizer.pad_token_id
        )
    
    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# --- Prompts ---
def build_simple_prompt(abstract, news_length):
    system = "You are an expert science journalist. Write a clear, engaging news article. Output ONLY the article text."
    user = f"ABSTRACT:\n{abstract}\n\nWrite a {news_length} word article grounded strictly in the abstract."
    return system, user

def build_drafter_prompt(abstract, news_length):
    system = "You are an expert science journalist. Draft an engaging news article. Output ONLY the draft text."
    user = f"ABSTRACT:\n{abstract}\n\nDraft a {news_length} word article based on the abstract."
    return system, user

def build_revision_prompt(abstract, draft):
    system = "You are a senior editor. Revise this draft for factual accuracy and hook. Output ONLY the final polished article."
    user = f"ORIGINAL ABSTRACT:\n{abstract}\n\nINITIAL DRAFT:\n{draft}\n\nRevise and output the final polished article."
    return system, user

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input .jsonl file")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B", help="Model path")
    ap.add_argument("--out", required=True, help="Output .jsonl file")
    ap.add_argument("--news_length", type=int, default=450)
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # 32B fits easily on one GPU or can be split. device_map="auto" will handle it.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa" # recommended for A100s
    )

    # Load data
    if not os.path.exists(args.data):
        print(f"Error: {args.data} not found.")
        return

    records = [json.loads(l) for l in open(args.data) if l.strip()]
    
    print(f"Starting generation for {len(records)} records...")
    with open(args.out, "a") as w:
        for idx, ex in enumerate(records):
            abstract, qas = ex["paper_abstract"], ex.get("qa_annotations", [])
            candidates = []

            # 2 Simple Samples
            for i in range(2):
                sys, usr = build_simple_prompt(abstract, args.news_length)
                art = generate_one(model, tokenizer, sys, usr, 800, 0.1)
                candidates.append({"source": f"simple_{i+1}", "article": art})

            # 2 Agentic Samples (Draft -> Revise)
            for i in range(2):
                sys, usr = build_drafter_prompt(abstract, args.news_length)
                draft = generate_one(model, tokenizer, sys, usr, 800, 0.5)
                
                sys_r, usr_r = build_revision_prompt(abstract, draft)
                art = generate_one(model, tokenizer, sys_r, usr_r, 800, 0.1)
                candidates.append({"source": f"agentic_{i+1}", "article": art})

            # Save results
            ex["candidates"] = candidates
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
            w.flush()
            print(f"[{idx+1}/{len(records)}] Article ID {ex.get('article_id', 'N/A')} - 4 candidates generated.")

if __name__ == "__main__":
    main()
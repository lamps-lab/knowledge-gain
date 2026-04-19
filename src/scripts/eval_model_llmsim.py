#!/usr/bin/env python3
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from llmsim_reward_cached import LLMSimKGRewardCached


def qas_to_text(qas):
    blocks = []
    for qa in qas or []:
        qid = qa.get("question_in_set")
        q = qa.get("question_text") or qa.get("question-text") or ""
        opts = "\n".join([f"  {i+1}. {o}" for i, o in enumerate(qa.get("options", []))])
        blocks.append(f"Q{qid}: {q}\n{opts}")
    return "\n\n".join(blocks)


def build_prompt(tokenizer, abstract):
    system = (
        "You are an expert science journalist. "
        "Write a clear, accurate, engaging science news article. "
        "Output ONLY the article text."
    )

    user = (
        f"ABSTRACT:\n{abstract}\n\n"
        f"Write a science news article grounded strictly in the abstract."
    )

    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_article(model, tokenizer, prompt, max_new_tokens=800):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_simulated_readers", type=int, default=5)
    ap.add_argument("--max_examples", type=int, default=300)
    ap.add_argument("--use_qas", action="store_true")
    ap.add_argument("--pre_cache", default="cache/pre_cache_eval.jsonl")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    scorer = LLMSimKGRewardCached(
        n_simulated_readers=args.n_simulated_readers,
        seed=123,
        pre_cache_path=args.pre_cache,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n = 0
    total_kg = 0.0
    total_pre = 0.0
    total_post = 0.0

    with open(args.data, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            if n >= args.max_examples:
                break

            ex = json.loads(line)
            abstract = ex["paper_abstract"]
            qas = ex["qa_annotations"]

            prompt = build_prompt(tokenizer, abstract)
            article = generate_article(model, tokenizer, prompt)

            score = scorer.score_article(
                abstract=abstract,
                article=article,
                qas=qas,
                return_details=False,
            )

            rec = {
                "article_id": ex.get("article_id"),
                "kg": score["kg"],
                "pre_acc": score["pre_acc"],
                "post_acc": score["post_acc"],
                "generated_article": article,
            }

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            w.flush()

            total_kg += score["kg"]
            total_pre += score["pre_acc"]
            total_post += score["post_acc"]
            n += 1

            print(
                f"[{n}] KG={score['kg']:.3f} "
                f"pre={score['pre_acc']:.3f} post={score['post_acc']:.3f}"
            )

    print("\nSUMMARY")
    print(f"N={n}")
    print(f"Mean KG={total_kg / max(1, n):.4f}")
    print(f"Mean PRE={total_pre / max(1, n):.4f}")
    print(f"Mean POST={total_post / max(1, n):.4f}")


if __name__ == "__main__":
    main()
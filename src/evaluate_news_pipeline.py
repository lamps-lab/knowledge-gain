#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SCORE_RE = re.compile(
    r"A\s*=\s*([1-5])\s*;\s*COMP\s*=\s*([1-5])\s*;\s*REL\s*=\s*([1-5])\s*;\s*CLA\s*=\s*([1-5])\s*;\s*KG\s*=\s*([1-5])",
    flags=re.IGNORECASE,
)


def qas_to_text(qas: List[Dict[str, Any]]) -> str:
    blocks = []
    for qa in qas or []:
        qid = qa.get("question_in_set")
        q = qa.get("question-text") or qa.get("question_text") or ""
        opts = qa.get("options") or []
        opt_lines = "\n".join([f"  {i+1}. {o}" for i, o in enumerate(opts)])
        blocks.append(f"Q{qid}: {q}\n{opt_lines}")
    return "\n\n".join(blocks)


def render_chat_prompt(tokenizer, system_text: str, user_text: str) -> str:
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_scores(text: str) -> Optional[Dict[str, int]]:
    if not text:
        return None
    m = SCORE_RE.search(text.strip())
    if not m:
        return None
    a, comp, rel, cla, kg = map(int, m.groups())
    return {"accuracy": a, "completeness": comp, "relevance": rel, "clarity": cla, "knowledge_gain": kg}


def build_eval_prompt(abstract: str, qas: List[Dict[str, Any]], article: str) -> Tuple[str, str]:
    system = (
        "You are a strict evaluator of grounded science news writing. "
        "Accuracy dominates. Unsupported numbers or named entities must strongly reduce Accuracy. "
        "If the article is about the wrong study/topic, set Accuracy=1. Return only the exact score line."
    )
    user = (
        "You are evaluating a science news article for a general audience.\n"
        "Score five dimensions from 1 to 5.\n\n"
        "Critical rules:\n"
        "- Accuracy dominates every other dimension.\n"
        "- If the article introduces a material unsupported number, year, unit, named entity, sample size, institution, journal, country, or misstates the study/topic, Accuracy must be 1 or 2.\n"
        "- If the article is on the wrong paper/topic, set Accuracy=1.\n"
        "- Do NOT give high Knowledge Gain to an article that is inaccurate.\n\n"
        "Rubric:\n"
        "- Accuracy: factual faithfulness to the abstract.\n"
        "- Completeness: covers the core result, main mechanism/framework, main caveat, and important quantitative details when present.\n"
        "- Relevance: emphasizes the most newsworthy and significant aspects of the work.\n"
        "- Clarity: easy for a general audience to follow without losing scientific meaning.\n"
        "- Knowledge Gain: after reading ONLY the candidate article, a high-school educated reader could answer the generated questions.\n\n"
        "Return EXACTLY one line in this format and nothing else:\n"
        "A=<1-5>;COMP=<1-5>;REL=<1-5>;CLA=<1-5>;KG=<1-5>\n\n"
        f"PAPER ABSTRACT:\n{abstract.strip()}\n\n"
        f"GENERATED QUESTIONS:\n{qas_to_text(qas)}\n\n"
        f"CANDIDATE ARTICLE:\n{article.strip()}"
    )
    return system, user


def generate_one(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
    prompt = render_chat_prompt(tokenizer, system_prompt, user_prompt)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    in_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.05,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    return tokenizer.decode(out[0, in_len:], skip_special_tokens=True).strip()


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = mean(values)
    s = pstdev(values) if len(values) > 1 else 0.0
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--judge_device", default="cuda:1")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=520)
    ap.add_argument("--judge_max_new_tokens", type=int, default=48)
    ap.add_argument("--news_length", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.data, split="train")
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    judge_tok = AutoTokenizer.from_pretrained(args.judge_model, use_fast=True, trust_remote_code=True)
    if judge_tok.pad_token is None:
        judge_tok.pad_token = judge_tok.eos_token
    judge_tok.padding_side = "left"

    dev = torch.device(args.device)
    jdev = torch.device(args.judge_device)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True, device_map={"": dev.index} if dev.type == "cuda" else None)
    model.eval()
    judge = AutoModelForCausalLM.from_pretrained(args.judge_model, torch_dtype=dtype, trust_remote_code=True, device_map={"": jdev.index} if jdev.type == "cuda" else None)
    judge.eval()

    metrics = defaultdict(list)
    with open(args.out, "w", encoding="utf-8") as w:
        for i, ex in enumerate(ds):
            abstract = ex["paper_abstract"]
            qas = ex.get("qa_annotations", [])
            system = (
                "You are a science news writer.\n"
                "Write a clear, accurate, engaging article for a general audience.\n"
                "Accuracy comes before style.\n"
                "Do NOT invent facts.\n"
                "If a number or named entity is not explicitly in the abstract, do not include it.\n"
                "Prefer short paragraphs and cautious wording.\n"
                "Output ONLY the article text.\n"
            )
            user = (
                f"ABSTRACT:\n{abstract.strip()}\n\n"
                f"Write ONE science news article for a general audience in about {args.news_length} words.\n"
                "Output ONLY the article text.\n"
            )
            article = generate_one(model, tok, system, user, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
            jsp, jup = build_eval_prompt(abstract, qas, article)
            raw = generate_one(judge, judge_tok, jsp, jup, args.judge_max_new_tokens, 0.0, 1.0, 0)
            scores = parse_scores(raw) or {"accuracy": 1, "completeness": 1, "relevance": 1, "clarity": 1, "knowledge_gain": 1}
            for k, v in scores.items():
                metrics[k].append(v)
            rec = {"idx": i, "article_id": ex.get("article_id", ex.get("id", i)), "scores": scores, "judge_raw": raw, "article": article}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}] {scores}", flush=True)

    print("Average scores ± std:")
    for k in ["accuracy", "completeness", "relevance", "clarity", "knowledge_gain"]:
        m, s = mean_std(metrics[k])
        print(f"  {k}: {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
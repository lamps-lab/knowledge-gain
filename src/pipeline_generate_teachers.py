#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time
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
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def parse_scores(text: str) -> Optional[Dict[str, int]]:
    if not text:
        return None
    m = SCORE_RE.search(text.strip())
    if not m:
        return None
    a, comp, rel, cla, kg = map(int, m.groups())
    return {"accuracy": a, "completeness": comp, "relevance": rel, "clarity": cla, "knowledge_gain": kg}


def centered_likert(x: int) -> float:
    return (float(x) - 3.0) / 2.0


def weighted_total(scores: Dict[str, int], acc_w: float, comp_w: float, rel_w: float, cla_w: float, kg_w: float) -> float:
    return (
        acc_w * centered_likert(scores["accuracy"]) +
        comp_w * centered_likert(scores["completeness"]) +
        rel_w * centered_likert(scores["relevance"]) +
        cla_w * centered_likert(scores["clarity"]) +
        kg_w * centered_likert(scores["knowledge_gain"])
    )


def build_simple_prompt(abstract: str, news_length: int) -> Tuple[str, str]:
    system = (
        "You are an expert science journalist. "
        "Write a clear, engaging science news article for a general audience. "
        "Output only the article text."
    )
    user = (
        f"Write a science news article of about {news_length} words based on this abstract.\n\n"
        f"{abstract}"
    )
    return system, user


def build_drafter_prompt(abstract: str, news_length: int) -> Tuple[str, str]:
    system = (
        "You are an expert science journalist. "
        "Your job is to take complex academic abstracts and turn them into engaging, "
        "accessible news articles for the general public. "
        "Create a catchy headline, keep the tone informative and exciting, and make the "
        "science easy to understand without dumbing it down. "
        "Output only the article text."
    )
    user = (
        f"Please draft a news article in about {news_length} words based on this abstract:\n\n"
        f"{abstract}"
    )
    return system, user


def build_revision_prompt(abstract: str, draft: str) -> Tuple[str, str]:
    system = (
        "You are a strict but fair senior editor at a top science magazine. "
        "Your job is to review article drafts to ensure they are factually accurate based "
        "on the original abstract, check for readability, and improve the narrative flow. "
        "Do not introduce new facts, statistics, or claims outside of the abstract. "
        "Output only the final polished article text."
    )
    user = f"""ORIGINAL ABSTRACT:
{abstract}

INITIAL DRAFT:
{draft}

Please revise and polish the draft. Ensure no scientific inaccuracies were introduced,
improve the hook, and output ONLY the final polished article including the headline.
"""
    return system, user


def build_judge_prompt(abstract: str, qas: List[Dict[str, Any]], article: str) -> Tuple[str, str]:
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


def judge_article(model, tokenizer, abstract: str, qas: List[Dict[str, Any]], article: str, max_new_tokens: int) -> Tuple[Optional[Dict[str, int]], str]:
    sp, up = build_judge_prompt(abstract, qas, article)
    raw = generate_one(model, tokenizer, sp, up, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, top_k=0)
    return parse_scores(raw), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gen_model", required=True)
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen_device", default="cuda:0")
    ap.add_argument("--judge_device", default="cuda:1")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=520)
    ap.add_argument("--judge_max_new_tokens", type=int, default=48)
    ap.add_argument("--news_length", type=int, default=320)
    ap.add_argument("--simple_samples", type=int, default=2)
    ap.add_argument("--agentic_samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=-1)
    ap.add_argument("--acc_weight", type=float, default=0.55)
    ap.add_argument("--comp_weight", type=float, default=0.15)
    ap.add_argument("--rel_weight", type=float, default=0.05)
    ap.add_argument("--cla_weight", type=float, default=0.10)
    ap.add_argument("--kg_weight", type=float, default=0.15)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = load_dataset("json", data_files=args.data, split="train")
    start = max(0, args.start)
    stop = len(ds) if args.stop < 0 else min(len(ds), args.stop)

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    gen_tok = AutoTokenizer.from_pretrained(args.gen_model, use_fast=True, trust_remote_code=True)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    gen_tok.padding_side = "left"

    judge_tok = AutoTokenizer.from_pretrained(args.judge_model, use_fast=True, trust_remote_code=True)
    if judge_tok.pad_token is None:
        judge_tok.pad_token = judge_tok.eos_token
    judge_tok.padding_side = "left"

    gen_dev = torch.device(args.gen_device)
    judge_dev = torch.device(args.judge_device)

    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": gen_dev.index} if gen_dev.type == "cuda" else None,
    )
    gen_model.eval()

    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": judge_dev.index} if judge_dev.type == "cuda" else None,
    )
    judge_model.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for idx in range(start, stop):
            ex = ds[idx]
            article_id = ex.get("article_id", ex.get("id", idx))
            abstract = ex["paper_abstract"]
            qas = ex.get("qa_annotations", [])

            candidates = []

            for sidx in range(args.simple_samples):
                sp, up = build_simple_prompt(abstract, args.news_length)
                art = generate_one(gen_model, gen_tok, sp, up, args.max_new_tokens, temperature=0.65, top_p=0.9, top_k=20)
                scores, judge_raw = judge_article(judge_model, judge_tok, abstract, qas, art, args.judge_max_new_tokens)
                total = weighted_total(scores, args.acc_weight, args.comp_weight, args.rel_weight, args.cla_weight, args.kg_weight) if scores else -9.0
                candidates.append({
                    "source": "simple",
                    "sample_idx": sidx,
                    "article": art,
                    "scores": scores,
                    "judge_raw": judge_raw,
                    "weighted_total": total,
                })

            for aidx in range(args.agentic_samples):
                sp, up = build_drafter_prompt(abstract, args.news_length)
                draft = generate_one(gen_model, gen_tok, sp, up, args.max_new_tokens, temperature=0.75, top_p=0.9, top_k=20)
                rsp, rup = build_revision_prompt(abstract, draft)
                art = generate_one(gen_model, gen_tok, rsp, rup, args.max_new_tokens, temperature=0.15, top_p=1.0, top_k=0)
                scores, judge_raw = judge_article(judge_model, judge_tok, abstract, qas, art, args.judge_max_new_tokens)
                total = weighted_total(scores, args.acc_weight, args.comp_weight, args.rel_weight, args.cla_weight, args.kg_weight) if scores else -9.0
                candidates.append({
                    "source": "agentic",
                    "sample_idx": aidx,
                    "article": art,
                    "scores": scores,
                    "judge_raw": judge_raw,
                    "weighted_total": total,
                })

            candidates.sort(key=lambda x: x["weighted_total"], reverse=True)
            best = candidates[0]
            row = {
                "article_id": article_id,
                "paper_abstract": abstract,
                "qa_annotations": qas,
                "best_source": best["source"],
                "best_article": best["article"],
                "best_scores": best["scores"],
                "best_weighted_total": best["weighted_total"],
                "all_candidates": candidates,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{idx}] article_id={article_id} best={best['source']} total={best['weighted_total']:.3f}", flush=True)


if __name__ == "__main__":
    main()
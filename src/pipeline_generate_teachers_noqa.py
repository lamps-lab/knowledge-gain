#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SCORE_RE = re.compile(
    r"A\s*=\s*([1-5])\s*;\s*COMP\s*=\s*([1-5])\s*;\s*REL\s*=\s*([1-5])\s*;\s*CLA\s*=\s*([1-5])\s*;\s*KG\s*=\s*([1-5])",
    flags=re.IGNORECASE,
)

NUM_TOKEN_RE = re.compile(
    r"(?<!\w)(?:\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?(?:[-–]\d+(?:\.\d+)?)?)(?:%|σ|×|x)?(?:/[A-Za-zµμ²0-9.\-]+)?"
)
MIXED_ALNUM_RE = re.compile(r"\b(?=\S*[A-Za-z])(?=\S*\d)[A-Za-z0-9.+/\-]+\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9.+/\-]*\b")
YEAR_RE = re.compile(r"\b(?:19\d{2}|20\d{2}|21\d{2})\b")


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
    return {
        "accuracy": a,
        "completeness": comp,
        "relevance": rel,
        "clarity": cla,
        "knowledge_gain": kg,
    }


def centered_likert(x: int) -> float:
    return (float(x) - 3.0) / 2.0


def weighted_total(
    scores: Dict[str, int],
    acc_w: float,
    comp_w: float,
    rel_w: float,
    cla_w: float,
    kg_w: float,
) -> float:
    return (
        acc_w * centered_likert(scores["accuracy"])
        + comp_w * centered_likert(scores["completeness"])
        + rel_w * centered_likert(scores["relevance"])
        + cla_w * centered_likert(scores["clarity"])
        + kg_w * centered_likert(scores["knowledge_gain"])
    )


def normalize_token(x: str) -> str:
    x = x.strip().replace("–", "-").replace("—", "-").replace("×", "x")
    x = x.replace(",", "")
    return x.lower()


def extract_numeric_tokens(text: str) -> Set[str]:
    return {normalize_token(m.group(0)) for m in NUM_TOKEN_RE.finditer(text or "")}


def extract_entityish_tokens(text: str) -> Set[str]:
    toks = {normalize_token(m.group(0)) for m in MIXED_ALNUM_RE.finditer(text or "")}
    toks.update({normalize_token(m.group(0)) for m in ACRONYM_RE.finditer(text or "")})
    return {t for t in toks if any(ch.isalpha() for ch in t)}


def grounding_score(article: str, abstract: str) -> float:
    article_nums = extract_numeric_tokens(article)
    abstract_nums = extract_numeric_tokens(abstract)
    extra_nums = article_nums - abstract_nums
    shared_nums = article_nums & abstract_nums

    article_ents = extract_entityish_tokens(article)
    abstract_ents = extract_entityish_tokens(abstract)
    extra_ents = article_ents - abstract_ents
    shared_ents = article_ents & abstract_ents

    article_years = {normalize_token(x) for x in YEAR_RE.findall(article or "")}
    abstract_years = {normalize_token(x) for x in YEAR_RE.findall(abstract or "")}
    extra_years = article_years - abstract_years

    score = 0.0
    score -= min(1.40, 0.28 * len(extra_nums))
    score -= min(0.90, 0.18 * len(extra_ents))
    score += min(0.25, 0.03 * len(shared_nums))
    score += min(0.20, 0.02 * len(shared_ents))
    score -= min(0.80, 0.25 * len(extra_years))

    if len(extra_nums) >= 2:
        score -= 0.35
    if len(extra_nums) + len(extra_ents) >= 4:
        score -= 0.35

    return float(score)


def light_format_score(text: str, min_words: int, max_words: int) -> float:
    t = (text or "").strip()
    if not t:
        return -2.0

    words = t.split()
    wc = len(words)
    chars = [ch for ch in t if not ch.isspace()]
    if not chars:
        return -2.0

    alpha = sum(ch.isalpha() for ch in chars) / len(chars)
    punct = sum(ch in string.punctuation for ch in chars) / len(chars)

    score = 0.0
    if min_words <= wc <= max_words:
        score += 0.15
    elif wc < (min_words - 50) or wc > (max_words + 120):
        score -= 0.55
    else:
        score -= 0.15

    if alpha < 0.60:
        score -= 0.40
    if punct > 0.32:
        score -= 0.30
    if "the abstract" in t.lower():
        score -= 0.35

    return float(score)


def get_abstract(ex: Dict[str, Any]) -> str:
    return str(ex.get("paper_abstract") or ex.get("abstract") or "").strip()


def build_simple_prompt(abstract: str, min_words: int, max_words: int) -> Tuple[str, str]:
    system = (
        "You are a top-tier science journalist and editor. "
        "Write a clear, accurate, engaging science news article for a general audience. "
        "Accuracy comes before style. Do NOT invent facts, names, institutions, numbers, dates, or publication details. "
        "Output only the article text."
    )
    user = (
        f"ABSTRACT:\n{abstract.strip()}\n\n"
        "Write ONE science news article for a general audience.\n"
        "Output ONLY the article text.\n\n"
        "Requirements:\n"
        "- Start with a punchy headline on the first line (no period).\n"
        f"- Length: {min_words}-{max_words} words.\n"
        "- Structure: 4-7 short paragraphs separated by blank lines.\n"
        "- Lead with the main finding in plain language.\n"
        "- Explain why the finding matters.\n"
        "- Cover the core result, main mechanism/framework, and main caveat if present.\n"
        "- Every factual claim must be supported by the abstract.\n"
        "- If a number, year, unit, score, sample size, institution, journal name, country, author name, or specific named method is not explicitly in the abstract, do NOT include it.\n"
        "- If you include a number from the abstract, copy it exactly.\n"
        "- Prefer cautious wording unless an exact number is both present and important.\n"
        "- Briefly explain technical terms for non-specialists.\n"
        "- Do NOT mention 'the abstract' or that you were given an abstract.\n"
        "- Keep the tone journalistic, concrete, and not hypey.\n"
    )
    return system, user


def build_revision_prompt(abstract: str, draft: str) -> Tuple[str, str]:
    system = (
        "You are a strict senior science editor. "
        "Revise the draft so it is fully grounded in the abstract, easier to understand, and more informative. "
        "Do NOT introduce any unsupported numbers, names, institutions, dates, methods, or claims. "
        "Output only the final article text."
    )
    user = f"""ORIGINAL ABSTRACT:
{abstract}

INITIAL DRAFT:
{draft}

Revise the draft.
Requirements:
- Preserve only facts supported by the abstract
- Improve clarity and information density
- Keep a headline
- Keep the article concrete, cautious, and readable
- Output ONLY the final article text
"""
    return system, user


def build_judge_prompt_noqa(abstract: str, article: str) -> Tuple[str, str]:
    system = (
        "You are a strict evaluator of grounded science news writing. "
        "Accuracy dominates. Unsupported numbers or named entities must strongly reduce Accuracy. "
        "If the article is about the wrong study/topic, set Accuracy=1. "
        "Return only the exact score line."
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
        "- Relevance: emphasizes the most newsworthy aspects of the work.\n"
        "- Clarity: easy for a general audience to follow without losing scientific meaning.\n"
        "- Knowledge Gain: after reading ONLY the candidate article, a reader should understand what was studied, what was found, why it matters, how it works or was studied, and the main caveat if present.\n\n"
        "Return EXACTLY one line in this format and nothing else:\n"
        "A=<1-5>;COMP=<1-5>;REL=<1-5>;CLA=<1-5>;KG=<1-5>\n\n"
        f"PAPER ABSTRACT:\n{abstract.strip()}\n\n"
        f"CANDIDATE ARTICLE:\n{article.strip()}"
    )
    return system, user


def generate_one(model, tokenizer, system_prompt: str, user_prompt: str,
                 max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
    prompt = render_chat_prompt(tokenizer, system_prompt, user_prompt)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    in_len = enc["input_ids"].shape[1]

    do_sample = temperature > 0
    gen_kwargs = dict(
        **enc,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.05,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k

    out = model.generate(**gen_kwargs)
    return tokenizer.decode(out[0, in_len:], skip_special_tokens=True).strip()


def judge_article(
    model,
    tokenizer,
    abstract: str,
    article: str,
    max_new_tokens: int,
) -> Tuple[Optional[Dict[str, int]], str]:
    sp, up = build_judge_prompt_noqa(abstract, article)
    raw = generate_one(
        model,
        tokenizer,
        sp,
        up,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )
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
    ap.add_argument("--min_words", type=int, default=220)
    ap.add_argument("--max_words", type=int, default=420)

    ap.add_argument("--simple_samples", type=int, default=3)
    ap.add_argument("--agentic_samples", type=int, default=0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=-1)

    ap.add_argument("--acc_weight", type=float, default=0.55)
    ap.add_argument("--comp_weight", type=float, default=0.20)
    ap.add_argument("--rel_weight", type=float, default=0.10)
    ap.add_argument("--cla_weight", type=float, default=0.10)
    ap.add_argument("--kg_weight", type=float, default=0.05)

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
    if hasattr(judge_model, "generation_config") and judge_model.generation_config is not None:
        judge_model.generation_config.do_sample = False
        judge_model.generation_config.temperature = None
        judge_model.generation_config.top_p = None
        judge_model.generation_config.top_k = None
    judge_model.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as w:
        for idx in range(start, stop):
            ex = ds[idx]
            article_id = ex.get("article_id", ex.get("id", idx))
            abstract = get_abstract(ex)
            if not abstract:
                continue

            candidates = []

            for sidx in range(args.simple_samples):
                sp, up = build_simple_prompt(abstract, args.min_words, args.max_words)
                art = generate_one(
                    gen_model, gen_tok, sp, up,
                    args.max_new_tokens,
                    temperature=0.45, top_p=0.90, top_k=0,
                )
                scores, judge_raw = judge_article(
                    judge_model, judge_tok, abstract, art, args.judge_max_new_tokens
                )
                judge_total = weighted_total(
                    scores,
                    args.acc_weight,
                    args.comp_weight,
                    args.rel_weight,
                    args.cla_weight,
                    args.kg_weight,
                ) if scores else -9.0
                grounding = grounding_score(art, abstract) + light_format_score(art, args.min_words, args.max_words)
                total = judge_total + grounding

                candidates.append({
                    "source": "simple",
                    "sample_idx": sidx,
                    "article": art,
                    "scores": scores,
                    "judge_raw": judge_raw,
                    "judge_total": judge_total,
                    "grounding": grounding,
                    "weighted_total": total,
                })

            for aidx in range(args.agentic_samples):
                sp, up = build_simple_prompt(abstract, args.min_words, args.max_words)
                draft = generate_one(
                    gen_model, gen_tok, sp, up,
                    args.max_new_tokens,
                    temperature=0.55, top_p=0.92, top_k=0,
                )
                rsp, rup = build_revision_prompt(abstract, draft)
                art = generate_one(
                    gen_model, gen_tok, rsp, rup,
                    args.max_new_tokens,
                    temperature=0.0, top_p=1.0, top_k=0,
                )
                scores, judge_raw = judge_article(
                    judge_model, judge_tok, abstract, art, args.judge_max_new_tokens
                )
                judge_total = weighted_total(
                    scores,
                    args.acc_weight,
                    args.comp_weight,
                    args.rel_weight,
                    args.cla_weight,
                    args.kg_weight,
                ) if scores else -9.0
                grounding = grounding_score(art, abstract) + light_format_score(art, args.min_words, args.max_words)
                total = judge_total + grounding

                candidates.append({
                    "source": "agentic",
                    "sample_idx": aidx,
                    "article": art,
                    "scores": scores,
                    "judge_raw": judge_raw,
                    "judge_total": judge_total,
                    "grounding": grounding,
                    "weighted_total": total,
                })

            candidates.sort(key=lambda x: x["weighted_total"], reverse=True)
            best = candidates[0]

            accepted = (
                best["scores"] is not None
                and best["scores"]["accuracy"] >= 4
                and best["scores"]["completeness"] >= 4
                and best["scores"]["clarity"] >= 4
                and best["grounding"] >= -0.15
            )

            row = {
                "article_id": article_id,
                "paper_abstract": abstract,
                "best_source": best["source"],
                "best_article": best["article"],
                "best_scores": best["scores"],
                "best_grounding": best["grounding"],
                "best_weighted_total": best["weighted_total"],
                "accepted_for_sft": accepted,
                "all_candidates": candidates,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{idx}] article_id={article_id} "
                f"best={best['source']} total={best['weighted_total']:.3f} "
                f"grounding={best['grounding']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
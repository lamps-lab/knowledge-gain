#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KG_AI evaluation with open-source LLMs:
- Simulate 10 (configurable) stochastic pre/post annotations per question set
- Build Q(KG) and compare to P(KG) from human annotations via sum of KL divergences

Usage examples:
  # HuggingFace Transformers locally (GPU recommended)
  python kg_ai_eval.py \
    --data-root . \
    --in grouped_docs=kgain_docs_grouped.json \
    --model-backend hf \
    --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --runs 10 --temperature 0.9 --top-p 0.95 --max-new-tokens 600

  # vLLM/OpenAI-compatible HTTP endpoint (open-source model served elsewhere)
  python kg_ai_eval.py \
    --data-root . \
    --in grouped_docs=kgain_docs_grouped.json \
    --model-backend openai \
    --openai-base-url http://localhost:8000/v1 \
    --openai-api-key sk-noop \
    --model-name llama-3.1-8b-instruct \
    --runs 10 --temperature 0.9 --top-p 0.95 --max-new-tokens 600

Outputs:
  - kg_ai_runs.jsonl         # all AI runs (per article/media, per repetition) with pre/post answers + KG
  - kg_ai_kl_summary.json    # per-(article, media) P/Q hists and D_KL, plus global totals
  - kg_ai_kl_summary.csv     # same summary as CSV
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PROMPT_PRE_SYSTEM = (
    "You are a test-taker. Use only your general knowledge. "
    "If unsure, select “I do not know the answer”. Do not invent facts. No explanations needed."
)

PROMPT_PRE_USER_FMT = """Output a strict JSON array, where each item must be:
{{
  "question-id": <int>,
  "question": "<verbatim question text>",
  "answer": "<verbatim option text>" | "I do not know the answer.",
  "confidence": <int 0-100>,
  "grounding": "general" | "unknown"
}}

Rules:
- If your confidence is below 50, set "answer": "I do not know the answer." and "grounding":"unknown".
- Do not use any tools, or external documents.
- Treat questions independently, do not carry information between them.

Questions:
{questions_block}
"""

PROMPT_POST_SYSTEM = (
    "You are a careful reader and test-taker. First, read the [MEDIA] and then answer the multiple choice questions "
    "using only information justified from the [MEDIA]. If the [MEDIA] is not sufficient, select “I do not know the answer.”"
)

PROMPT_POST_USER_FMT = """Output a strict JSON array, where each item must be:
{{
  "question-id": <int>,
  "question": "<verbatim question text>",
  "answer": "<verbatim option text>" | "I do not know the answer.",
  "confidence": <int 0-100>,
  "grounding": "media" | "unknown",
  "evidence": "<up to 12 consecutive words from the [MEDIA] OR ''>"
}}
Rules:
- Base the answers from the media you read.
- If not justified, set "answer": "I do not know the answer.", "grounding": "unknown", and "evidence": "".

[MEDIA]
{media}

Questions:
{questions_block}
"""

# --------------------------
# Utilities
# --------------------------

DK_LABEL = "I do not know the answer."

def nrm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()

def ensure_dk_last(options: List[str]) -> List[str]:
    """Ensure DK exists and is last."""
    has_dk = any(nrm(o) == nrm(DK_LABEL) or nrm(o) in {"i do not know", "i dont know", "i don't know", "dk", "idk"} for o in options)
    opts = [o for o in options if nrm(o) != nrm(DK_LABEL)]
    if not has_dk:
        opts.append(DK_LABEL)
    else:
        # move canonical DK last (remove any variant first)
        opts = [o for o in opts if nrm(o) != nrm(DK_LABEL)]
        opts.append(DK_LABEL)
    return opts

def option_index(answer_text: Optional[str], options: List[str]) -> int:
    """Return 1-based index of answer; default to DK if unmapped/None."""
    if not answer_text:
        return len(options)
    ans = nrm(answer_text)
    for i, o in enumerate(options, start=1):
        if nrm(o) == ans:
            return i
    # heuristics: if the answer starts with "A)"/"1)" etc., try strip
    m = re.match(r"^\s*([A-J]|[1-9]\d*)\s*[\)\].:]\s*(.*)$", answer_text, re.IGNORECASE)
    if m:
        residue = m.group(2).strip()
        for i, o in enumerate(options, start=1):
            if nrm(o) == nrm(residue):
                return i
    # fallback → DK
    return len(options)

def json_sanitize_to_array(txt: str) -> List[Dict[str, Any]]:
    """Extract the first JSON array from the text. Fall back to empty list."""
    # Find the first [ ... ] balanced block
    start = txt.find("[")
    end = txt.rfind("]")
    if start >= 0 and end > start:
        chunk = txt[start:end+1]
        try:
            arr = json.loads(chunk)
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
    # Last resort: try to repair common trailing commas
    try:
        txt2 = re.sub(r",(\s*[\]\}])", r"\1", txt[start:end+1])
        arr = json.loads(txt2)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    return []

def safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

# --------------------------
# Model client backends
# --------------------------

@dataclass
class GenConfig:
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 600

class ModelClient:
    def generate(self, system_prompt: str, user_prompt: str, cfg: GenConfig) -> str:
        raise NotImplementedError

class HFClient(ModelClient):
    def __init__(self, model_name: str, device: Optional[str] = None, dtype: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto" if device is None else device)
        self.pipe = pipeline("text-generation", model=model, tokenizer=tok)

    def generate(self, system_prompt: str, user_prompt: str, cfg: GenConfig) -> str:
        prompt = f"<s>[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n[USER]\n{user_prompt}\n[/USER]\n"
        out = self.pipe(
            prompt,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            eos_token_id=self.pipe.tokenizer.eos_token_id,
        )[0]["generated_text"]
        # Heuristic: return only the assistant fragment after our prompt
        return out[len(prompt):]

class OpenAICompatClient(ModelClient):
    """
    Works with vLLM/OpenAI-compatible servers (e.g., local vLLM hosting an open-source model).
    """
    def __init__(self, model_name: str, base_url: str, api_key: str):
        import requests  # noqa
        self.model = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate(self, system_prompt: str, user_prompt: str, cfg: GenConfig) -> str:
        import requests
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_new_tokens,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

# --------------------------
# Building the Questions block
# --------------------------

def build_questions_block(qa_list: List[Dict[str, Any]]) -> str:
    """
    Provide a JSON array that contains for each item:
      - question-id: 1..6 (local within this set)
      - question
      - options (exact, DK last)
    This appears in the *prompt payload*; the model's output must be a strict JSON array (per instructions).
    """
    items = []
    for qa in qa_list:
        options = ensure_dk_last(list(qa["options"]))
        items.append({
            "question-id": int(qa["question_in_set"]),
            "question": str(qa["question-text"]),
            "options": options,
        })
    return json.dumps(items, ensure_ascii=False, indent=2)

# --------------------------
# KG and distributions
# --------------------------

def count_correct(qa_list: List[Dict[str, Any]], which: str) -> int:
    """which ∈ {'human-answer-pre','human-answer-post','ai-pre','ai-post'}"""
    total = 0
    for qa in qa_list:
        correct_idx = qa.get("correct_option")
        if not isinstance(correct_idx, int):
            continue
        pick = qa.get(which)
        if isinstance(pick, int) and pick == correct_idx:
            total += 1
    return total

def kg_hist_from_humans(human_annots: List[Dict[str, Any]]) -> Dict[int, int]:
    """Return histogram over KG = post - pre (support -6..+6) from human annotations."""
    hist = {k: 0 for k in range(-6, 7)}
    for ann in human_annots:
        qa = ann["qa_annotations"]
        pre = count_correct(qa, "human-answer-pre")
        post = count_correct(qa, "human-answer-post")
        kg = post - pre
        hist[kg] += 1
    return hist

def kg_hist_from_ai(ai_runs: List[Tuple[int, int]]) -> Dict[int, int]:
    """Input list of (pre_correct, post_correct) pairs."""
    hist = {k: 0 for k in range(-6, 7)}
    for pre_c, post_c in ai_runs:
        kg = post_c - pre_c
        hist[kg] += 1
    return hist

def normalize_hist(hist: Dict[int, int], eps: float = 1e-6) -> Dict[int, float]:
    total = sum(hist.values())
    if total == 0:
        # Smooth to uniform tiny mass if really empty (shouldn't happen)
        return {k: eps for k in hist}
    return {k: max(v / total, eps) for k, v in hist.items()}

def dkl_pq(p: Dict[int, float], q: Dict[int, float]) -> float:
    s = 0.0
    for k in sorted(p.keys()):
        pk, qk = p[k], q[k]
        s += pk * math.log(pk / qk)
    return float(s)

# --------------------------
# AI answering (one run)
# --------------------------

def run_ai_once_for_set(
    client: ModelClient,
    qa_list: List[Dict[str, Any]],
    media: str,
    cfg: GenConfig,
    rng: random.Random,
) -> Tuple[List[int], List[int], str, str]:
    """
    Returns:
      pre_idxs: list of chosen option indices (1-based) for 6 Qs
      post_idxs: same after reading media
      raw_pre_text, raw_post_text: raw model outputs (for audit)
    """
    questions_block = build_questions_block(qa_list)

    # PRE (no media)
    pre_user = PROMPT_PRE_USER_FMT.format(questions_block=questions_block)
    pre_raw = client.generate(PROMPT_PRE_SYSTEM, pre_user, cfg)
    pre_arr = json_sanitize_to_array(pre_raw)

    # POST (with media)
    post_user = PROMPT_POST_USER_FMT.format(media=media, questions_block=questions_block)
    post_raw = client.generate(PROMPT_POST_SYSTEM, post_user, cfg)
    post_arr = json_sanitize_to_array(post_raw)

    # Build local map question-id -> options
    id2opts: Dict[int, List[str]] = {}
    for qa in qa_list:
        opts = ensure_dk_last(list(qa["options"]))
        id2opts[int(qa["question_in_set"])] = opts

    def arr_to_indices(arr) -> List[int]:
        # expected 6 items; resilient: map by question-id; default DK
        out = []
        by_id = {}
        for it in arr:
            try:
                qid = int(it.get("question-id"))
            except Exception:
                continue
            by_id[qid] = it

        for qa in qa_list:
            qid = int(qa["question_in_set"])
            opts = id2opts[qid]
            ans_text = None
            if qid in by_id:
                ans_text = by_id[qid].get("answer")
            out.append(option_index(ans_text, opts))
        return out

    pre_idxs = arr_to_indices(pre_arr)
    post_idxs = arr_to_indices(post_arr)
    return pre_idxs, post_idxs, pre_raw, post_raw

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=".", help="Dir containing input JSON")
    ap.add_argument("--in", dest="in_path", type=str, default="kgain_docs_grouped.json",
                    help="Grouped doc JSON (with human_annotations)")
    ap.add_argument("--out-prefix", type=str, default="kg_ai",
                    help="Output prefix (default: kg_ai)")
    ap.add_argument("--runs", type=int, default=10, help="Stochastic runs per set")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1234)

    # Model backend config
    ap.add_argument("--model-backend", choices=["hf", "openai"], default="hf")
    ap.add_argument("--model-name", type=str, required=True,
                    help="HF model id or OpenAI-compatible model name")
    ap.add_argument("--hf-device", type=str, default=None)
    ap.add_argument("--hf-dtype", type=str, default=None)
    ap.add_argument("--openai-base-url", type=str, default=None)
    ap.add_argument("--openai-api-key", type=str, default=None)

    args = ap.parse_args()
    data_root = Path(args.data_root)
    in_path = data_root / args.in_path

    # Load dataset
    docs = json.loads(Path(in_path).read_text(encoding="utf-8"))

    # Init model client
    if args.model_backend == "hf":
        try:
            client = HFClient(args.model_name, device=args.hf_device, dtype=args.hf_dtype)
        except Exception as e:
            raise RuntimeError(f"Failed to init HF model '{args.model_name}': {e}")
    else:
        base = args.openai_base_url or os.environ.get("OPENAI_BASE_URL", "")
        key = args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not base or not key:
            raise RuntimeError("--openai-base-url and --openai-api-key are required for model-backend=openai")
        client = OpenAICompatClient(args.model_name, base, key)

    cfg = GenConfig(temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
    rng = random.Random(args.seed)

    # Outputs
    runs_path = data_root / f"{args.out_prefix}_runs.jsonl"
    summ_json_path = data_root / f"{args.out_prefix}_kl_summary.json"
    summ_csv_path = data_root / f"{args.out_prefix}_kl_summary.csv"

    # Per-(article, media) rows for summary
    summary_rows: List[Dict[str, Any]] = []
    global_dkl_sum = 0.0
    total_sets = 0

    with runs_path.open("w", encoding="utf-8") as fw:
        for doc in docs:
            article_id = doc["article-id"]
            media = doc["content-type"]
            content = doc.get("content") or ""
            human_annots = doc.get("human_annotations", [])
            if not human_annots:
                # still process AI but P will be empty; we skip DKL in that case
                pass

            # Build the fixed QA list (6 questions) from the first annotator entry
            # safer: get from doc by aligning unique question_in_set to their meta
            qa_template_map = {}
            for ann in human_annots:
                for qa in ann["qa_annotations"]:
                    qi = int(qa["question_in_set"])
                    if qi not in qa_template_map:
                        # freeze the metadata; ensure DK last
                        qa_copy = dict(qa)
                        qa_copy["options"] = ensure_dk_last(list(qa["options"]))
                        qa_template_map[qi] = qa_copy
                if len(qa_template_map) == 6:
                    break
            qa_list = [qa_template_map[i] for i in sorted(qa_template_map.keys())]
            if len(qa_list) != 6:
                # fallback: skip malformed doc
                continue

            # Human P(KG)
            P_hist = kg_hist_from_humans(human_annots)
            P_prob = normalize_hist(P_hist) if sum(P_hist.values()) > 0 else None

            # AI runs
            ai_runs_pairs: List[Tuple[int, int]] = []
            for r in range(args.runs):
                pre_idxs, post_idxs, pre_raw, post_raw = run_ai_once_for_set(
                    client=client, qa_list=qa_list, media=content, cfg=cfg, rng=rng
                )
                # Count correctness for AI
                # Attach to qa_list temporarily to reuse count_correct
                for i, qa in enumerate(qa_list):
                    qa["ai-pre"] = pre_idxs[i]
                    qa["ai-post"] = post_idxs[i]
                pre_correct = count_correct(qa_list, "ai-pre")
                post_correct = count_correct(qa_list, "ai-post")
                ai_runs_pairs.append((pre_correct, post_correct))

                # Write raw run for audit
                fw.write(json.dumps({
                    "article-id": article_id,
                    "content-type": media,
                    "run": r + 1,
                    "pre_correct": pre_correct,
                    "post_correct": post_correct,
                    "kg": post_correct - pre_correct,
                    "pre_raw": pre_raw,
                    "post_raw": post_raw,
                }, ensure_ascii=False) + "\n")

            # Q(KG)
            Q_hist = kg_hist_from_ai(ai_runs_pairs)
            Q_prob = normalize_hist(Q_hist)

            # D_KL (only if P is available)
            dkl = None
            if P_prob is not None:
                dkl = dkl_pq(P_prob, Q_prob)
                global_dkl_sum += dkl
                total_sets += 1

            # Save per-set summary row
            row = {
                "article-id": article_id,
                "content-type": media,
                **{f"P_KG_{k}": v for k, v in P_hist.items()},
                **{f"Q_KG_{k}": v for k, v in Q_hist.items()},
                "P_total": sum(P_hist.values()),
                "Q_total": sum(Q_hist.values()),
                "DKL_P_Q": dkl,
            }
            summary_rows.append(row)

    # Save summaries
    with summ_json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "global_sum_DKL": global_dkl_sum,
            "num_sets_with_P": total_sets,
            "per_set": summary_rows,
        }, f, ensure_ascii=False, indent=2)

    pd.DataFrame(summary_rows).to_csv(summ_csv_path, index=False)

    print(f"[DONE] Wrote AI runs to {runs_path}")
    print(f"[DONE] Wrote KL summary to {summ_json_path} and {summ_csv_path}")
    if total_sets > 0:
        print(f"Global Σ DKL(P||Q) over {total_sets} sets = {global_dkl_sum:.4f}")
    else:
        print("No human P distributions found; computed Q only.")

if __name__ == "__main__":
    main()

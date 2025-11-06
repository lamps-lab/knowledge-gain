import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prompts
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

# Utils
DK_LABEL = "I do not know the answer."

def nrm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()

def ensure_dk_last(options: List[str]) -> List[str]:
    has_dk = any(nrm(o) == nrm(DK_LABEL) or nrm(o) in {"i do not know","i dont know","i don't know","dk","idk"} for o in options)
    opts = [o for o in options if nrm(o) != nrm(DK_LABEL)]
    if not has_dk:
        opts.append(DK_LABEL)
    else:
        opts = [o for o in opts if nrm(o) != nrm(DK_LABEL)]
        opts.append(DK_LABEL)
    return opts

def option_index(answer_text: Optional[str], options: List[str]) -> int:
    if not answer_text:
        return len(options)
    ans = nrm(answer_text)
    for i, o in enumerate(options, start=1):
        if nrm(o) == ans:
            return i
    m = re.match(r"^\s*([A-J]|[1-9]\d*)\s*[\)\].:]\s*(.*)$", answer_text, re.IGNORECASE)
    if m:
        residue = m.group(2).strip()
        for i, o in enumerate(options, start=1):
            if nrm(o) == nrm(residue):
                return i
    return len(options)

def json_sanitize_to_array(txt: str) -> List[Dict[str, Any]]:
    start = txt.find("["); end = txt.rfind("]")
    if start >= 0 and end > start:
        chunk = txt[start:end+1]
        try:
            arr = json.loads(chunk)
            if isinstance(arr, list): return arr
        except Exception:
            try:
                chunk = re.sub(r",(\s*[\]\}])", r"\1", chunk)
                arr = json.loads(chunk)
                if isinstance(arr, list): return arr
            except Exception:
                pass
    return []

def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-_.")

def build_questions_block(qa_list: List[Dict[str, Any]]) -> str:
    items = []
    for qa in qa_list:
        options = ensure_dk_last(list(qa["options"]))
        items.append({
            "question-id": int(qa["question_in_set"]),
            "question": str(qa["question-text"]),
            "options": options,
        })
    return json.dumps(items, ensure_ascii=False, indent=2)

# KG + KL
def count_correct(qa_list: List[Dict[str, Any]], which: str) -> int:
    total = 0
    for qa in qa_list:
        correct_idx = qa.get("correct_option")
        pick = qa.get(which)
        if isinstance(correct_idx, int) and isinstance(pick, int) and pick == correct_idx:
            total += 1
    return total

def kg_hist_from_humans(human_annots: List[Dict[str, Any]]) -> Dict[int, int]:
    hist = {k: 0 for k in range(-6, 7)}
    for ann in human_annots:
        qa = ann["qa_annotations"]
        pre = count_correct(qa, "human-answer-pre")
        post = count_correct(qa, "human-answer-post")
        hist[post - pre] += 1
    return hist

def kg_hist_from_ai(ai_runs: List[Tuple[int, int]]) -> Dict[int, int]:
    hist = {k: 0 for k in range(-6, 7)}
    for pre_c, post_c in ai_runs:
        hist[post_c - pre_c] += 1
    return hist

def normalize_hist(hist: Dict[int, int], eps: float = 1e-6) -> Dict[int, float]:
    total = sum(hist.values())
    if total == 0: return {k: eps for k in hist}
    return {k: max(v / total, eps) for k, v in hist.items()}

def dkl_pq(p: Dict[int, float], q: Dict[int, float]) -> float:
    return float(sum(p[k] * math.log(p[k] / q[k]) for k in sorted(p)))

# Perf helpers 
def to_torch_dtype(name: Optional[str]):
    if not name or name.lower() == "auto": return "auto"
    name = name.lower()
    if name in ("bfloat16","bf16"): return torch.bfloat16
    if name in ("float16","fp16"):  return torch.float16
    if name in ("float32","fp32"):  return torch.float32
    return "auto"

def truncate_to_tokens(text: str, tok: AutoTokenizer, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return text
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tok.decode(ids, skip_special_tokens=False)

# Minimal chat generation
def chat_once(model, tok, system_prompt: str, user_prompt: str,
              temperature: float, top_p: float, max_new_tokens: int) -> str:
    # Build a string template first, then tokenize (so we can create an attention_mask)
    if getattr(tok, "chat_template", None):
        prompt_text = tok.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            add_generation_prompt=True,
            tokenize=False  # return string, we re-tokenize to get mask
        )
    else:
        prompt_text = f"<s>[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n[USER]\n{user_prompt}\n[/USER]\n"

    enc = tok(
        prompt_text,
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_attention_mask=True,
    )

    input_ids = enc["input_ids"]
    attn_mask = enc.get("attention_mask", torch.ones_like(input_ids))

    # put inputs on the first device the model expects
    dev = next(iter(set(p.device for p in model.parameters())), torch.device("cpu"))
    input_ids = input_ids.to(dev)
    attn_mask = attn_mask.to(dev)

    eos_id = tok.eos_token_id or getattr(model.config, "eos_token_id", None)
    pad_id = tok.pad_token_id or getattr(model.config, "pad_token_id", None) or eos_id

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
        )
    new_tokens = out[0, input_ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)

def run_ai_once_for_set(model, tok, qa_list: List[Dict[str, Any]], media: str,
                        temperature: float, top_p: float, max_new_tokens: int) -> Tuple[List[int], List[int], str, str]:
    questions_block = build_questions_block(qa_list)

    pre_user = PROMPT_PRE_USER_FMT.format(questions_block=questions_block)
    pre_raw = chat_once(model, tok, PROMPT_PRE_SYSTEM, pre_user, temperature, top_p, max_new_tokens)
    pre_arr = json_sanitize_to_array(pre_raw)

    post_user = PROMPT_POST_USER_FMT.format(media=media, questions_block=questions_block)
    post_raw = chat_once(model, tok, PROMPT_POST_SYSTEM, post_user, temperature, top_p, max_new_tokens)
    post_arr = json_sanitize_to_array(post_raw)

    id2opts = {int(qa["question_in_set"]): ensure_dk_last(list(qa["options"])) for qa in qa_list}

    def arr_to_indices(arr) -> List[int]:
        by_id = {}
        for it in arr:
            try: qid = int(it.get("question-id"))
            except Exception: continue
            by_id[qid] = it
        out = []
        for qa in qa_list:
            qid = int(qa["question_in_set"])
            ans_text = by_id.get(qid, {}).get("answer")
            out.append(option_index(ans_text, id2opts[qid]))
        return out

    return arr_to_indices(pre_arr), arr_to_indices(post_arr), pre_raw, post_raw

# Main 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=".", help="Dir containing input JSON")
    ap.add_argument("--in", dest="in_path", type=str, default="kgain_docs_grouped.json",
                    help="Grouped doc JSON (with human_annotations)")
    ap.add_argument("--out-prefix", type=str, default="kg_ai", help="Base output prefix (model name appended)")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--hf-dtype", type=str, default="auto",
                    choices=["auto","bfloat16","float16","float32","bf16","fp16","fp32"])
    ap.add_argument("--attn-impl", type=str, default="sdpa",
                    choices=["sdpa","flash_attention_2","eager"],
                    help="Attention backend if your stack supports it.")
    ap.add_argument("--media-max-tokens", type=int, default=2000,
                    help="Truncate [MEDIA] to this many tokens for speed; 0 disables truncation.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # load dataset
    data_root = Path(args.data_root)
    docs = json.loads((data_root / args.in_path).read_text(encoding="utf-8"))

    # tokenizer (try fast first, then fallback)
    try:
        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only models prefer left padding for batched gen

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=to_torch_dtype(args.hf_dtype),
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    )
    model.eval()

    # outputs include model name
    model_slug = slugify(args.model_name)
    prefix = f"{args.out_prefix}__{model_slug}"
    runs_path = data_root / f"{prefix}_runs.jsonl"
    summ_json_path = data_root / f"{prefix}_kl_summary.json"
    summ_csv_path = data_root / f"{prefix}_kl_summary.csv"

    summary_rows: List[Dict[str, Any]] = []
    global_dkl_sum = 0.0
    total_sets = 0

    # small helper to persist after every write
    def _flush(fh):
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass

    with runs_path.open("w", encoding="utf-8") as fw:
        for doc in docs:
            article_id = doc["article-id"]
            media = doc["content-type"]
            content = doc.get("content") or ""
            human_annots = doc.get("human_annotations", [])

            # fixed QA list (6)
            qa_template_map = {}
            for ann in human_annots:
                for qa in ann["qa_annotations"]:
                    qi = int(qa["question_in_set"])
                    if qi not in qa_template_map:
                        qa_copy = dict(qa)
                        qa_copy["options"] = ensure_dk_last(list(qa["options"]))
                        qa_template_map[qi] = qa_copy
                if len(qa_template_map) == 6:
                    break
            qa_list = [qa_template_map[i] for i in sorted(qa_template_map.keys())]
            if len(qa_list) != 6:
                continue

            # human P(KG)
            P_hist = kg_hist_from_humans(human_annots)
            P_prob = normalize_hist(P_hist) if sum(P_hist.values()) > 0 else None

            # Truncate MEDIA for speed
            media_text = truncate_to_tokens(content, tok, args.media_max_tokens)

            # AI runs
            ai_pairs: List[Tuple[int, int]] = []
            for r in range(args.runs):
                pre_idxs, post_idxs, pre_raw, post_raw = run_ai_once_for_set(
                    model=model, tok=tok, qa_list=qa_list, media=media_text,
                    temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens
                )
                for i, qa in enumerate(qa_list):
                    qa["ai-pre"] = pre_idxs[i]
                    qa["ai-post"] = post_idxs[i]
                pre_correct = count_correct(qa_list, "ai-pre")
                post_correct = count_correct(qa_list, "ai-post")
                ai_pairs.append((pre_correct, post_correct))

                rec = {
                    "article-id": article_id,
                    "content-type": media,
                    "model-name": args.model_name,
                    "run": r + 1,
                    "pre_correct": pre_correct,
                    "post_correct": post_correct,
                    "kg": post_correct - pre_correct,
                    "pre_raw": pre_raw,
                    "post_raw": post_raw,
                }
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _flush(fw)  # <-- persist each run

            Q_hist = kg_hist_from_ai(ai_pairs)
            Q_prob = normalize_hist(Q_hist)

            dkl = None
            if P_prob is not None:
                dkl = dkl_pq(P_prob, Q_prob)
                global_dkl_sum += dkl
                total_sets += 1

            summary_rows.append({
                "article-id": article_id,
                "content-type": media,
                **{f"P_KG_{k}": v for k, v in P_hist.items()},
                **{f"Q_KG_{k}": v for k, v in Q_hist.items()},
                "P_total": sum(P_hist.values()),
                "Q_total": sum(Q_hist.values()),
                "DKL_P_Q": dkl,
            })

    with summ_json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "global_sum_DKL": global_dkl_sum,
            "num_sets_with_P": total_sets,
            "per_set": summary_rows,
        }, f, ensure_ascii=False, indent=2)

    pd.DataFrame(summary_rows).to_csv(summ_csv_path, index=False)

    # quick visibility on sharding/offload (if any)
    if hasattr(model, "hf_device_map"):
        print("[device_map]", model.hf_device_map)

    print(f"[DONE] Wrote AI runs to {runs_path}")
    print(f"[DONE] Wrote KL summary to {summ_json_path} and {summ_csv_path}")
    if total_sets > 0:
        print(f"Global Σ DKL(P||Q) over {total_sets} sets = {global_dkl_sum:.4f}")
    else:
        print("No human P distributions found; computed Q only.")

if __name__ == "__main__":
    main()


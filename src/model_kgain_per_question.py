#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# keep TF quiet and avoid pulling TF graph stuff
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

# ---------------- Prompts (single-question) ----------------

PROMPT_PRE_SYSTEM = (
    "You are a test-taker. Use only your general knowledge. "
    "If unsure, select “I do not know the answer”. Do not invent facts. No explanations needed."
)

# Ask ONE question at a time; demand a single JSON object (no prose, no code fences)
PROMPT_PRE_ONE_FMT = """Return ONLY a JSON object with exactly these keys:
{{
  "question-id": <int>,
  "question": "<verbatim question text>",
  "answer": "<verbatim option text>" | "I do not know the answer.",
  "confidence": <int 0-100>,
  "grounding": "general" | "unknown"
}}
Rules:
- If your confidence is below 50, set "answer": "I do not know the answer." and "grounding":"unknown".
- Do not add extra keys. Do not include "options" in your output. No code fences. No prose.

Question (with options for you to choose from):
{question_block}
"""

PROMPT_POST_SYSTEM = (
    "You are a careful reader and test-taker. First, read the [MEDIA] and then answer the multiple choice question "
    "using only information justified from the [MEDIA]. If the [MEDIA] is not sufficient, select “I do not know the answer.”"
)

PROMPT_POST_ONE_FMT = """Return ONLY a JSON object with exactly these keys:
{{
  "question-id": <int>,
  "question": "<verbatim question text>",
  "answer": "<verbatim option text>" | "I do not know the answer.",
  "confidence": <int 0-100>,
  "grounding": "media" | "unknown",
  "evidence": "<up to 12 consecutive words from the [MEDIA] OR ''>"
}}
Rules:
- Base the answer only on the [MEDIA]. If the [MEDIA] is not sufficient, answer: "I do not know the answer.", set "grounding":"unknown", "evidence": "".
- Do not add extra keys. Do not include "options" in your output. No code fences. No prose.

[MEDIA]
{media}

Question (with options for you to choose from):
{question_block}
"""

# ---------------- Utils ----------------

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
    """Map the model's answer text to a 1-based option index. If unknown/IDK, return len(options)."""
    if not answer_text:
        return len(options)
    ans = nrm(answer_text)

    # direct match
    for i, o in enumerate(options, start=1):
        if nrm(o) == ans:
            return i

    # e.g., "A) blah", "1. blah", etc.
    m = re.match(r"^\s*([A-J]|[1-9]\d*)\s*[\)\].:]\s*(.*)$", answer_text, re.IGNORECASE)
    if m:
        residue = m.group(2).strip()
        for i, o in enumerate(options, start=1):
            if nrm(o) == nrm(residue):
                return i

    # fallback: if answer text is a prefix of one option (rare but helpful)
    for i, o in enumerate(options, start=1):
        if nrm(o).startswith(ans) or ans.startswith(nrm(o)):
            return i

    # default to "I don't know" bucket (last)
    return len(options)

def _strip_code_fences(txt: str) -> str:
    # remove ```json ... ``` or ``` ... ```
    txt = re.sub(r"^```(?:json)?\s*", "", txt.strip(), flags=re.IGNORECASE)
    txt = re.sub(r"\s*```$", "", txt.strip())
    return txt

def _extract_first_json_obj(txt: str) -> Optional[str]:
    """
    Grab the first balanced {...} object from text. Tolerant to junk around it.
    Also handles a one-element array by unwrapping to its first element.
    """
    s = _strip_code_fences(txt)
    start = s.find("{")
    if start < 0:
        # sometimes the model returns a one-element array; handle that transparently
        if "[" in s and "]" in s:
            try:
                arr = json.loads(s[s.find("["):s.rfind("]")+1])
                if isinstance(arr, list) and arr:
                    return json.dumps(arr[0], ensure_ascii=False)
            except Exception:
                pass
        return None
    # simple stack-based brace matching
    depth, end = 0, None
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None
    return s[start:end+1]

def parse_one_json_obj(txt: str) -> Dict[str, Any]:
    """Tolerant single-object JSON parser."""
    cand = _extract_first_json_obj(txt)
    if not cand:
        return {}
    try:
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}

def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-_.")

# ------------- KG + KL --------------

def count_correct(qa_list: List[Dict[str, Any]], which: str) -> int:
    total = 0
    for qa in qa_list:
        correct_idx = qa.get("correct_option")
        pick = qa.get(which)
        if isinstance(correct_idx, int) and isinstance(pick, int) and pick == correct_idx:
            total += 1
    return total

def kg_hist_from_humans(human_annots: List[Dict[str, Any]]) -> Dict[int, int]:
    # 6 questions → KG in [-6..+6]
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

# ------------- Perf helpers -------------

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

# ------------- Minimal chat generation (with attention_mask) -------------

def chat_once(model, tok, system_prompt: str, user_prompt: str,
              temperature: float, top_p: float, max_new_tokens: int) -> str:
    """
    NOTE: This generates ONE completion for ONE prompt.
    We call this function separately for pre and post, and once per question.
    """
    # Build a string template first, then tokenize (so we can create an attention_mask)
    if getattr(tok, "chat_template", None):
        prompt_text = tok.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            add_generation_prompt=True,
            tokenize=False  # return string; we re-tokenize to craft attention_mask
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

    # put inputs on the model's device
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    input_ids = input_ids.to(dev)
    attn_mask = attn_mask.to(dev)

    eos_id = tok.eos_token_id or getattr(model.config, "eos_token_id", None)
    pad_id = tok.pad_token_id or getattr(model.config, "pad_token_id", None) or eos_id

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,  # avoid the pad==eos ambiguity warning
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

# ------------- One-question runner -------------

def question_block_for_prompt(qid: int, qtext: str, options: List[str]) -> str:
    data = {
        "question-id": int(qid),
        "question": str(qtext),
        "options": list(options),
    }
    return json.dumps(data, ensure_ascii=False, indent=2)

def answer_one(
    stage: str,  # "pre" or "post"
    model,
    tok,
    qid: int,
    qtext: str,
    options: List[str],
    media_text: Optional[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Tuple[int, Dict[str, Any], str]:
    """Ask one question; return (index, parsed_obj, raw_text)."""
    qblock = question_block_for_prompt(qid, qtext, options)
    if stage == "pre":
        user = PROMPT_PRE_ONE_FMT.format(question_block=qblock)
        raw = chat_once(model, tok, PROMPT_PRE_SYSTEM, user, temperature, top_p, max_new_tokens)
    else:
        assert stage == "post"
        user = PROMPT_POST_ONE_FMT.format(media=media_text or "", question_block=qblock)
        raw = chat_once(model, tok, PROMPT_POST_SYSTEM, user, temperature, top_p, max_new_tokens)

    obj = parse_one_json_obj(raw)

    # try to read answer from a few common fields
    ans_text = None
    for k in ("answer", "final_answer", "choice", "pred", "prediction"):
        if isinstance(obj.get(k), str) and obj[k].strip():
            ans_text = obj[k]
            break

    idx = option_index(ans_text, options)
    return idx, obj, raw

def run_ai_once_for_set(
    model,
    tok,
    qa_list: List[Dict[str, Any]],
    media_text: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    log_prefix: str,
) -> Tuple[List[int], List[int], List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str]]:
    pre_idxs, post_idxs = [], []
    pre_objs, post_objs = [], []
    pre_raws, post_raws = [], []

    # PRE: one question per fresh context
    for qa in qa_list:
        qid = int(qa["question_in_set"])
        qtext = str(qa["question-text"])
        opts = ensure_dk_last(list(qa["options"]))
        t0 = time.time()
        idx, obj, raw = answer_one(
            stage="pre",
            model=model, tok=tok,
            qid=qid, qtext=qtext, options=opts, media_text=None,
            temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens
        )
        dt = time.time() - t0
        pre_idxs.append(idx)
        pre_objs.append(obj)
        pre_raws.append(raw)
        print(f"{log_prefix} PRE Q{qid}: idx={idx}/{len(opts)} in {dt:.2f}s"); sys.stdout.flush()

    # POST: again one question at a time, but include [MEDIA]
    for qa in qa_list:
        qid = int(qa["question_in_set"])
        qtext = str(qa["question-text"])
        opts = ensure_dk_last(list(qa["options"]))
        t0 = time.time()
        idx, obj, raw = answer_one(
            stage="post",
            model=model, tok=tok,
            qid=qid, qtext=qtext, options=opts, media_text=media_text,
            temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens
        )
        dt = time.time() - t0
        post_idxs.append(idx)
        post_objs.append(obj)
        post_raws.append(raw)
        print(f"{log_prefix} POST Q{qid}: idx={idx}/{len(opts)} in {dt:.2f}s"); sys.stdout.flush()

    return pre_idxs, post_idxs, pre_objs, post_objs, pre_raws, post_raws

# ---------------- Main ----------------

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
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # load dataset
    data_root = Path(args.data_root)
    docs = json.loads((data_root / args.in_path).read_text(encoding="utf-8"))

    # tokenizer (try fast first, fallback to slow)
    try:
        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    # ensure pad token (pad==eos is fine if we pass attention_mask)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only models prefer left padding for batched gen

    # model (attempt with requested attention impl; fallback if unsupported)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=to_torch_dtype(args.hf_dtype),
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=args.attn_impl,
        )
    except TypeError:
        # older Transformers may not support attn_implementation
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=to_torch_dtype(args.hf_dtype),
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
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
    global_kg_sum_running = 0.0  # running KG sum across all runs/docs

    # small helper to persist after every write
    def _flush(fh):
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass

    total_docs = len(docs)
    doc_counter = 0
    global_runs_done = 0
    global_runs_total = len(docs) * args.runs

    t_all0 = time.time()
    with runs_path.open("w", encoding="utf-8") as fw:
        for doc in docs:
            doc_counter += 1
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
                print(f"[SKIP] {article_id}: only {len(qa_list)} questions found.")
                sys.stdout.flush()
                continue

            # human P(KG)
            P_hist = kg_hist_from_humans(human_annots)
            P_prob = normalize_hist(P_hist) if sum(P_hist.values()) > 0 else None

            # Truncate MEDIA for speed
            media_text = truncate_to_tokens(content, tok, args.media_max_tokens)

            print(f"\n=== Doc {doc_counter}/{total_docs} | {article_id} [{media}] ===")
            sys.stdout.flush()

            # AI runs
            ai_pairs: List[Tuple[int, int]] = []
            run_pre_acc_sum = 0
            run_post_acc_sum = 0
            run_kg_sum = 0.0
            t_doc0 = time.time()

            for r in range(args.runs):
                run_prefix = f"[{article_id} r{r+1}/{args.runs}]"
                t0 = time.time()
                pre_idxs, post_idxs, pre_objs, post_objs, pre_raws, post_raws = run_ai_once_for_set(
                    model=model, tok=tok, qa_list=qa_list, media_text=media_text,
                    temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                    log_prefix=run_prefix
                )

                # attach to qa_list for scoring
                for i, qa in enumerate(qa_list):
                    qa["ai-pre"] = pre_idxs[i]
                    qa["ai-post"] = post_idxs[i]

                pre_correct = count_correct(qa_list, "ai-pre")
                post_correct = count_correct(qa_list, "ai-post")
                kg_val = post_correct - pre_correct

                ai_pairs.append((pre_correct, post_correct))
                run_pre_acc_sum += pre_correct
                run_post_acc_sum += post_correct
                run_kg_sum += kg_val
                global_runs_done += 1
                global_kg_sum_running += kg_val
                dt = time.time() - t0

                # running stats
                runs_done_here = r + 1
                pre_mean_here = run_pre_acc_sum / runs_done_here
                post_mean_here = run_post_acc_sum / runs_done_here
                kg_mean_here = run_kg_sum / runs_done_here
                global_kg_mean = global_kg_sum_running / max(global_runs_done, 1)

                print(f"{run_prefix} DONE  pre={pre_correct}/6  post={post_correct}/6  kg={kg_val:+d}  "
                      f"run_time={dt:.2f}s | avg_pre={pre_mean_here:.2f} avg_post={post_mean_here:.2f} "
                      f"avg_kg={kg_mean_here:+.2f}  global {global_runs_done}/{global_runs_total} "
                      f"global_kg_mean={global_kg_mean:+.2f}")
                sys.stdout.flush()

                # persist after each run
                rec = {
                    "article-id": article_id,
                    "content-type": media,
                    "model-name": args.model_name,
                    "run": r + 1,
                    "pre_correct": pre_correct,
                    "post_correct": post_correct,
                    "kg": kg_val,
                    "pre_indices": pre_idxs,
                    "post_indices": post_idxs,
                    "pre_objs": pre_objs,
                    "post_objs": post_objs,
                    "pre_raw_items": pre_raws,
                    "post_raw_items": post_raws,
                    "avg_pre_so_far": pre_mean_here,
                    "avg_post_so_far": post_mean_here,
                    "avg_kg_so_far": kg_mean_here,
                    "global_kg_mean_so_far": global_kg_mean
                }
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _flush(fw)

            # per-document KL vs human P(KG)
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

            print(f"[DOC DONE] {article_id} in {time.time()-t_doc0:.2f}s")
            sys.stdout.flush()

    # write summaries
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

    print(f"\n[DONE] Wrote AI runs to {runs_path}")
    print(f"[DONE] Wrote KL summary to {summ_json_path} and {summ_csv_path}")
    if total_sets > 0:
        print(f"Global Σ DKL(P||Q) over {total_sets} sets = {global_dkl_sum:.4f}")
    else:
        print("No human P distributions found; computed Q only.")
    print(f"[TOTAL ELAPSED] {time.time()-t_all0:.2f}s")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
train_grpo_qwen3_kgain.py

GRPO fine-tuning of a generator to maximize a trained reward model.

Data JSONL must contain:
  {"paper_abstract": "...", "qa_annotations":[...]}

Generator sees only abstract.
Reward model sees abstract + questions + candidate article.

Recommended:
- Generator on cuda:1
- Reward model on cuda:0
- LoRA enabled
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional, Tuple

import torch

# TRL/PyTorch FSDP compatibility shim (some torch builds lack FSDPModule, TRL imports it).
try:
    import torch.distributed.fsdp as fsdp
    if not hasattr(fsdp, "FSDPModule") and hasattr(fsdp, "FullyShardedDataParallel"):
        fsdp.FSDPModule = fsdp.FullyShardedDataParallel
except Exception:
    pass

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer


IDK_TEXT = "I do not know the answer."


def qas_to_text(qas: List[Dict[str, Any]]) -> str:
    blocks = []
    for qa in qas:
        qid = qa.get("question_in_set")
        q = qa.get("question-text") or qa.get("question_text") or ""
        opts = qa.get("options") or []
        opt_lines = "\n".join([f"  {i+1}. {o}" for i, o in enumerate(opts)])
        blocks.append(f"Q{qid}: {q}\n{opt_lines}")
    return "\n\n".join(blocks)


def build_generator_prompt(abstract: str) -> str:
    return (
        "You are a science news writer.\n"
        "Write a clear, engaging news article for a general audience.\n"
        "Length: 450–750 words.\n"
        "Stop when the article is complete. Do NOT add filler or extra sections.\n"
        "Do NOT invent facts: no made-up numbers, years, institutions, author names, or journal names.\n"
        "Only include quantitative results if they are explicitly present in the abstract.\n"
        "Prefer short paragraphs and a journalistic tone.\n\n"
        f"ABSTRACT:\n{abstract.strip()}\n\n"
        "Write the news article now.\n"
    )


def build_rm_prompt(abstract: str, qas: List[Dict[str, Any]]) -> str:
    return (
        "You are scoring a candidate science news article.\n"
        "Score higher if it would increase a typical reader's ability to answer the questions after reading.\n\n"
        f"PAPER ABSTRACT:\n{abstract.strip()}\n\n"
        f"QUESTION SET:\n{qas_to_text(qas)}\n\n"
        "CANDIDATE CONTENT:\n"
    )


def alpha_ratio(text: str) -> float:
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    alpha = sum(c.isalpha() for c in non_space)
    return alpha / len(non_space)


def length_reward(words: int, min_words: int, max_words: int, weight: float) -> float:
    # Penalize being outside the word range. Inside the range gives 0.
    if words < min_words:
        return -weight * (min_words - words)
    if words > max_words:
        return -weight * (words - max_words)
    return 0.0


def alpha_reward(ar: float, min_alpha: float, weight: float) -> float:
    # Penalize low alphabetic ratio (math/garbage tends to have low ratio).
    if ar >= min_alpha:
        return 0.0
    return -weight * (min_alpha - ar)


@torch.no_grad()
def make_rm_reward_func(
    rm_model,
    rm_tokenizer,
    device: torch.device,
    max_length: int,
    microbatch: int,
    # reward shaping knobs
    min_words: int,
    max_words: int,
    len_weight: float,
    min_alpha: float,
    alpha_weight: float,
):
    rm_model.eval()

    def reward_func(completions, rm_prompt, **kwargs):
        texts = [p + c for p, c in zip(rm_prompt, completions)]
        mb = max(1, int(microbatch))
        rm_scores: List[float] = []

        for i in range(0, len(texts), mb):
            batch_texts = texts[i : i + mb]
            enc = rm_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            with torch.inference_mode():
                out = rm_model(**enc)

            rm_scores.extend(out.logits.squeeze(-1).float().detach().cpu().tolist())

        shaped: List[float] = []
        for s, c in zip(rm_scores, completions):
            w = len((c or "").split())
            ar = alpha_ratio(c or "")
            r_len = length_reward(w, min_words=min_words, max_words=max_words, weight=len_weight)
            r_alpha = alpha_reward(ar, min_alpha=min_alpha, weight=alpha_weight)
            shaped.append(float(s) + float(r_len) + float(r_alpha))

        return shaped

    return reward_func


class SampleGenerationCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        sample_record: Dict[str, Any],
        out_path: str,
        every_steps: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        print_samples: bool,
    ):
        self.tokenizer = tokenizer
        self.sample_record = sample_record
        self.out_path = out_path
        self.every = max(0, int(every_steps))
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.print_samples = bool(print_samples)

    def on_step_end(self, args, state, control, **kwargs):
        if self.every <= 0:
            return control
        if state.global_step == 0 or (state.global_step % self.every) != 0:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        try:
            prompt = self.sample_record["prompt"]
            article_id = self.sample_record.get("article_id")
            paper_abs = self.sample_record.get("paper_abstract")

            model_was_cache = getattr(model.config, "use_cache", None)
            model.config.use_cache = True

            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            device = next(model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            input_len = enc["input_ids"].shape[-1]

            with torch.inference_mode():
                out_ids = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            completion_ids = out_ids[0, input_len:]
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            wc = len(completion.split())
            ar = alpha_ratio(completion)

            row = {
                "step": int(state.global_step),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "article_id": article_id,
                "words": wc,
                "alpha_ratio": ar,
                "prompt": prompt,
                "paper_abstract": paper_abs,
                "completion": completion,
            }
            with open(self.out_path, "a", encoding="utf-8") as w:
                w.write(json.dumps(row, ensure_ascii=False) + "\n")

            if self.print_samples:
                preview = completion[:900].replace("\n", " ")
                print(f"\n[SAMPLE @ step {state.global_step}] words={wc} alpha={ar:.3f} :: {preview}...\n")

            if model_was_cache is not None:
                model.config.use_cache = model_was_cache

        except Exception as e:
            print(f"[SAMPLE CALLBACK ERROR @ step {state.global_step}] {repr(e)}")

        return control


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="rl_records.jsonl")
    ap.add_argument("--gen_model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--rm_model", required=True, help="path or HF id of trained RM")
    ap.add_argument("--out", default="qwen3_kgain_grpo")

    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--max_rm_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=900)

    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--num_generations", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--rm_device", default="cuda:0")
    ap.add_argument("--rm_microbatch", type=int, default=1)

    ap.add_argument("--gen_device", default="cuda:1")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--resume_from_checkpoint", default=None)

    ap.add_argument("--log_file", default=None, help="If set, redirects stdout/stderr to this file")

    ap.add_argument("--sample_every", type=int, default=100)
    ap.add_argument("--sample_out", default="samples_real.jsonl")
    ap.add_argument("--sample_temp", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)
    ap.add_argument("--sample_max_new_tokens", type=int, default=900)
    ap.add_argument("--sample_print", action="store_true", help="Also print sample preview to stdout")

    # Reward shaping guardrails (prevents junk that hacks the RM)
    ap.add_argument("--min_words", type=int, default=450)
    ap.add_argument("--max_words", type=int, default=750)
    ap.add_argument("--len_weight", type=float, default=0.01, help="Penalty per word outside target range")
    ap.add_argument("--min_alpha_ratio", type=float, default=0.60)
    ap.add_argument("--alpha_weight", type=float, default=10.0, help="Penalty scale for low alpha ratio")

    args = ap.parse_args()

    log_fh = None
    if args.log_file:
        log_fh = open(args.log_file, "a", encoding="utf-8")
        try:
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass

    ctx_stdout = redirect_stdout(log_fh) if log_fh else nullcontext()
    ctx_stderr = redirect_stderr(log_fh) if log_fh else nullcontext()

    with ctx_stdout, ctx_stderr:
        ds = load_dataset("json", data_files=args.data, split="train")

        gen_tok = AutoTokenizer.from_pretrained(args.gen_model, use_fast=True, trust_remote_code=True)
        if gen_tok.pad_token is None:
            gen_tok.pad_token = gen_tok.eos_token

        def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
            if max_tokens is None or max_tokens <= 0:
                return text
            ids = tokenizer(text, truncation=True, max_length=max_tokens, add_special_tokens=False).input_ids
            return tokenizer.decode(ids, skip_special_tokens=True)

        def _map(ex):
            abstract = ex["paper_abstract"]
            qas = ex["qa_annotations"]
            prompt = build_generator_prompt(abstract)
            ex["prompt"] = truncate_to_tokens(gen_tok, prompt, args.max_prompt_length)
            ex["rm_prompt"] = build_rm_prompt(abstract, qas)
            return ex

        ds = ds.map(_map)

        rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model, use_fast=True, trust_remote_code=True)
        if rm_tokenizer.pad_token is None:
            rm_tokenizer.pad_token = rm_tokenizer.eos_token

        rm_model = AutoModelForSequenceClassification.from_pretrained(
            args.rm_model,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        rm_device = torch.device(args.rm_device) if args.rm_device else torch.device("cpu")
        rm_model.to(rm_device)

        reward_func = make_rm_reward_func(
            rm_model=rm_model,
            rm_tokenizer=rm_tokenizer,
            device=rm_device,
            max_length=args.max_rm_length,
            microbatch=args.rm_microbatch,
            min_words=args.min_words,
            max_words=args.max_words,
            len_weight=args.len_weight,
            min_alpha=args.min_alpha_ratio,
            alpha_weight=args.alpha_weight,
        )

        gen_device = torch.device(args.gen_device)
        dtype = torch.bfloat16 if args.bf16 else torch.float16
        device_map = {"": gen_device.index} if gen_device.type == "cuda" else None

        gen_model = AutoModelForCausalLM.from_pretrained(
            args.gen_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=device_map,
        )

        gen_model.config.use_cache = False
        gen_model.gradient_checkpointing_enable()

        if args.use_lora:
            lora_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            gen_model = get_peft_model(gen_model, lora_cfg)
            gen_model.print_trainable_parameters()

        cfg = GRPOConfig(
            output_dir=args.out,
            learning_rate=args.lr,
            max_steps=args.steps,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_generations=args.num_generations,
            beta=args.beta,
            max_completion_length=args.max_new_tokens,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            remove_unused_columns=False,
        )

        trainer = GRPOTrainer(
            model=gen_model,
            args=cfg,
            train_dataset=ds,
            reward_funcs=reward_func,
        )

        if args.sample_every and args.sample_every > 0:
            sample_record = ds[0]
            trainer.add_callback(
                SampleGenerationCallback(
                    tokenizer=gen_tok,
                    sample_record=sample_record,
                    out_path=args.sample_out,
                    every_steps=args.sample_every,
                    max_new_tokens=args.sample_max_new_tokens,
                    temperature=args.sample_temp,
                    top_p=args.sample_top_p,
                    print_samples=args.sample_print,
                )
            )

        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(args.out)
        print(f"Saved GRPO-tuned generator to: {args.out}")

    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()
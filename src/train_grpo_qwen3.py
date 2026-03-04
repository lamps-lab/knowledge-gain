#!/usr/bin/env python3
"""
train_grpo_qwen3_kgain.py

RL fine-tuning (GRPO) of a generator to maximize the reward model.

Input RL records JSONL (from build_rm_and_rl_data_from_annotated.py):
  {"paper_abstract": "...", "qa_annotations":[...]}

Example:
  pip install "torch>=2.2" transformers datasets trl accelerate peft
  python train_grpo_qwen3_kgain.py \
    --data rl_records.jsonl \
    --gen_model Qwen/Qwen3-4B-Base \
    --rm_model rm_qwen3_kgain \
    --out qwen3_kgain_grpo \
    --num_generations 4 \
    --beta 0.02 \
    --steps 2000
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

# ---- TRL/PyTorch FSDP compatibility shim (must be BEFORE importing trl) ----
import torch

try:
    import torch.distributed.fsdp as fsdp
    # Some torch builds do not export FSDPModule; TRL expects it.
    # Map it to FullyShardedDataParallel so "from torch.distributed.fsdp import FSDPModule" works.
    if not hasattr(fsdp, "FSDPModule") and hasattr(fsdp, "FullyShardedDataParallel"):
        fsdp.FSDPModule = fsdp.FullyShardedDataParallel
except Exception:
    # If distributed/fsdp isn't available, we just skip — TRL will then rely on non-FSDP paths.
    pass
# --------------------------------------------------------------------------



from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import GRPOTrainer, GRPOConfig


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
    # Generator sees ONLY the abstract (prevents “writing to the test”).
    return (
        "You are a science news writer.\n"
        "Write a clear, engaging news article for a general audience.\n"
        "Do NOT invent facts: no made-up numbers, years, institutions, author names, or journal names.\n"
        "Only include quantitative results if they are explicitly present in the abstract.\n"
        "Prefer short paragraphs and a journalistic tone.\n\n"
        f"ABSTRACT:\n{abstract.strip()}\n\n"
        "Write the news article now.\n"
    )


def build_rm_prompt(abstract: str, qas: List[Dict[str, Any]]) -> str:
    # Reward model sees abstract + questions (defines the reward target).
    return (
        "You are scoring a candidate science news article.\n"
        "Score higher if it would increase a typical reader's ability to answer the questions after reading.\n\n"
        f"PAPER ABSTRACT:\n{abstract.strip()}\n\n"
        f"QUESTION SET:\n{qas_to_text(qas)}\n\n"
        "CANDIDATE CONTENT:\n"
    )


@torch.no_grad()
def make_rm_reward_func(rm_model, rm_tokenizer, device: torch.device, max_length: int):
    rm_model.eval()

    def reward_func(completions, rm_prompt, **kwargs):
        # TRL GRPO passes dataset columns as kwargs; rm_prompt is a list aligned with completions.
        texts = [p + c for p, c in zip(rm_prompt, completions)]
        enc = rm_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = rm_model(**enc)
        scores = out.logits.squeeze(-1).float().detach().cpu().tolist()
        return [float(s) for s in scores]

    return reward_func


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="rl_records.jsonl")
    ap.add_argument("--gen_model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--rm_model", required=True, help="path or HF id of trained RM")
    ap.add_argument("--out", default="qwen3_kgain_grpo")

    ap.add_argument("--max_prompt_length", type=int, default=1024)   # generator prompt
    ap.add_argument("--max_rm_length", type=int, default=4096)       # rm prompt + completion
    ap.add_argument("--max_new_tokens", type=int, default=900)

    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--beta", type=float, default=0.02, help="KL weight; 0 disables KL")
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.data, split="train")

    # Add generator prompt + rm_prompt columns
    def _map(ex):
        abstract = ex["paper_abstract"]
        qas = ex["qa_annotations"]
        ex["prompt"] = build_generator_prompt(abstract)
        ex["rm_prompt"] = build_rm_prompt(abstract, qas)
        return ex

    ds = ds.map(_map)

    # Load RM
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model, use_fast=True, trust_remote_code=True)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token

    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rm_model.to(device)

    reward_func = make_rm_reward_func(rm_model, rm_tokenizer, device, max_length=args.max_rm_length)

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        max_steps=args.steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        beta=args.beta,
        #max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_new_tokens,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=args.gen_model,      # TRL can load from string; or pass a model object if you prefer
        args=cfg,
        train_dataset=ds,
        reward_funcs=reward_func,
    )

    trainer.train()
    trainer.save_model(args.out)
    print(f"Saved GRPO-tuned generator to: {args.out}")


if __name__ == "__main__":
    main()

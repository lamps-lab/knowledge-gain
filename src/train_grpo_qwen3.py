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

# TRL/PyTorch FSDP compatibility shim
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
#

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

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
def make_rm_reward_func(rm_model, rm_tokenizer, device: torch.device, max_length: int, microbatch: int = 1):
    rm_model.eval()

    def reward_func(completions, rm_prompt, **kwargs):
        # Build full RM inputs
        texts = [p + c for p, c in zip(rm_prompt, completions)]

        scores = []
        mb = max(1, int(microbatch))

        # Microbatch to reduce peak memory
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
            scores.extend(out.logits.squeeze(-1).float().detach().cpu().tolist())

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
    ap.add_argument("--rm_device", default=None, help="Where to run reward model: cuda:1, cuda:0, or cpu")
    ap.add_argument("--rm_microbatch", type=int, default=1, help="Microbatch for RM scoring to avoid OOM")
    ap.add_argument("--gen_device", default="cuda:1", help="Device for generator, e.g. cuda:1")
    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA for generator")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--bf16", action="store_true", help="Use bf16 for generator if available")


    args = ap.parse_args()
    #   Load RL dataset  
    ds = load_dataset("json", data_files=args.data, split="train")

    # We'll need a generator tokenizer for prompt truncation (soft cap)
    gen_tok = AutoTokenizer.from_pretrained(args.gen_model, use_fast=True, trust_remote_code=True)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token

    def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
        if max_tokens is None or max_tokens <= 0:
            return text
        ids = tokenizer(text, truncation=True, max_length=max_tokens, add_special_tokens=False).input_ids
        return tokenizer.decode(ids, skip_special_tokens=True)

    # Add generator prompt + rm_prompt columns
    def _map(ex):
        abstract = ex["paper_abstract"]
        qas = ex["qa_annotations"]
        prompt = build_generator_prompt(abstract)
        ex["prompt"] = truncate_to_tokens(gen_tok, prompt, args.max_prompt_length)
        ex["rm_prompt"] = build_rm_prompt(abstract, qas)
        return ex

    ds = ds.map(_map)

    #   Load Reward Model (RM)  
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model, use_fast=True, trust_remote_code=True)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token

    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    if args.rm_device is None:
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            rm_device = torch.device("cuda:0")  # default RM to cuda:0; generator default is cuda:1
        elif torch.cuda.is_available():
            rm_device = torch.device("cuda:0")
        else:
            rm_device = torch.device("cpu")
    else:
        rm_device = torch.device(args.rm_device)

    rm_model.to(rm_device)

    reward_func = make_rm_reward_func(
        rm_model, rm_tokenizer, rm_device, max_length=args.max_rm_length, microbatch=args.rm_microbatch
    )

    #   Load Generator explicitly on chosen device  
    gen_device = torch.device(args.gen_device)
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    # device_map forces placement; for cuda it expects an int device index
    device_map = None
    if gen_device.type == "cuda":
        device_map = {"": gen_device.index}

    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )

    # Memory savers for training
    gen_model.config.use_cache = False
    gen_model.gradient_checkpointing_enable()

    # LoRA
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

    #   GRPO Config  
    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        max_steps=args.steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        beta=args.beta,
        max_completion_length=args.max_new_tokens,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=gen_model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=reward_func,
    )

    trainer.train()
    trainer.save_model(args.out)
    print(f"Saved GRPO-tuned generator to: {args.out}")

if __name__ == "__main__":
    main()

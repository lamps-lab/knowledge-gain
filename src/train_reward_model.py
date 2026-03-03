#!/usr/bin/env python3
"""
train_reward_model_trl.py

Trains a reward model from preference pairs:
JSONL with columns: prompt, chosen, rejected

Example:
  pip install "torch>=2.2" transformers datasets trl accelerate peft
  python train_reward_model_trl.py --data rm_pairs.jsonl --model Qwen/Qwen3-4B-Base --out rm_qwen3_kgain
"""

from __future__ import annotations

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="rm_pairs.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--out", default="rm_qwen3_kgain")
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.data, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    cfg = RewardConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        seed=args.seed,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved RM to: {args.out}")


if __name__ == "__main__":
    main()
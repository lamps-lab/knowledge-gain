#!/usr/bin/env python3
import argparse
import json

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel
from trl import DPOTrainer


def load_rows(path, tokenizer):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)

            prompt_text = tokenizer.apply_chat_template(
                r["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )

            rows.append({
                "prompt": prompt_text,
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct")
    ap.add_argument("--sft_adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_policy = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_policy, args.sft_adapter, is_trainable=True)

    base_ref = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(base_ref, args.sft_adapter, is_trainable=False)

    rows = load_rows(args.pairs, tokenizer)
    dataset = Dataset.from_list(rows)
    split = dataset.train_test_split(test_size=0.05, seed=42)

    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,
        num_train_epochs=1,
        bf16=True,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        seed=42,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=train_args,
        beta=args.beta,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        max_prompt_length=2048,
        max_length=3072,
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    print(f"Saved DPO model to {args.out}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def render_chat_prompt(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def build_messages(abstract: str, article: str, min_words: int, max_words: int) -> List[Dict[str, str]]:
    system = (
        "You are a science news writer.\n"
        "Write a clear, accurate, engaging article for a general audience.\n"
        "Accuracy comes before style.\n"
        "Do NOT invent facts.\n"
        "If a number or named entity is not explicitly in the abstract, do not include it.\n"
        "Prefer short paragraphs and cautious wording.\n"
        "Output ONLY the article text.\n"
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
        "- Cover the core result, the main mechanism or framework, and the main caveat if present.\n"
        "- Every factual claim must be supported by the abstract.\n"
        "- If a number, year, unit, score, sample size, institution, journal name, country, author name, or specific named method is not explicitly in the abstract, do NOT include it.\n"
        "- If you include a number from the abstract, copy it exactly.\n"
        "- Prefer cautious qualitative wording unless an exact number is both present and important.\n"
        "- If the abstract is sparse, stay sparse rather than filling in context.\n"
        "- Briefly explain technical terms for non-specialists.\n"
        "- Do NOT mention 'the abstract' or that you were given an abstract.\n"
        "- Do NOT invent details.\n"
        "- Keep the tone journalistic, concrete, and not hypey.\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": article.strip()},
    ]


def tokenize_conversation_with_assistant_only_labels(
    tokenizer,
    messages: List[Dict[str, str]],
    max_length: int,
) -> Dict[str, List[int]]:
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Each example must end with an assistant message.")

    full_text = render_chat_prompt(tokenizer, messages, add_generation_prompt=False)
    prompt_text = render_chat_prompt(tokenizer, messages[:-1], add_generation_prompt=True)

    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )["input_ids"]

    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )["input_ids"]

    if full_ids[: len(prompt_ids)] != prompt_ids:
        alt_prompt_text = render_chat_prompt(tokenizer, messages[:-1], add_generation_prompt=False)
        alt_prompt_ids = tokenizer(
            alt_prompt_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]

        if full_ids[: len(alt_prompt_ids)] == alt_prompt_ids:
            prompt_ids = alt_prompt_ids
        else:
            assistant_text = (messages[-1]["content"] or "").strip()
            assistant_ids = tokenizer(
                assistant_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"]

            start_idx = 0
            if assistant_ids:
                found = False
                for i in range(max(0, len(full_ids) - len(assistant_ids) + 1)):
                    if full_ids[i : i + len(assistant_ids)] == assistant_ids:
                        start_idx = i
                        found = True
                        break
                if not found:
                    start_idx = min(len(full_ids), len(prompt_ids))
            prompt_ids = full_ids[:start_idx]

    labels = full_ids.copy()
    assistant_start = min(len(prompt_ids), len(labels))
    labels[:assistant_start] = [-100] * assistant_start

    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval_data", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--max_length", type=int, default=3072)
    ap.add_argument("--min_words", type=int, default=220)
    ap.add_argument("--max_words", type=int, default=420)

    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.data, split="train")

    eval_ds = None
    if args.eval_data:
        eval_ds = load_dataset("json", data_files=args.eval_data, split="train")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    if args.use_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    def preprocess(ex):
        messages = build_messages(
            abstract=ex["paper_abstract"],
            article=ex["target_article"],
            min_words=args.min_words,
            max_words=args.max_words,
        )
        return tokenize_conversation_with_assistant_only_labels(
            tokenizer=tok,
            messages=messages,
            max_length=args.max_length,
        )

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    if eval_ds is not None:
        eval_ds = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)

    def collate(features):
        input_features = [
            {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]
        batch = tok.pad(input_features, padding=True, return_tensors="pt")

        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            lab = f["labels"]
            pad_len = max_len - len(lab)
            labels.append(lab + ([-100] * pad_len))

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    has_eval = eval_ds is not None

    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if has_eval else None,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=not args.bf16,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        optim="adamw_torch",
        #evaluation_strategy="steps" if has_eval else "no",
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        data_collator=collate,
    )

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved SFT model to {args.out}")


if __name__ == "__main__":
    main()

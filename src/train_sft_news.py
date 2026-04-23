#!/usr/bin/env python3
import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)

import json
from datasets import Dataset

def tokenize_conversation(tokenizer, messages, max_length):
    # Masking logic: only train on assistant responses
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    tokenized_full = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)
    tokenized_prompt = tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)

    input_ids = tokenized_full["input_ids"]
    labels = list(input_ids)
    
    prompt_len = len(tokenized_prompt["input_ids"])
    for i in range(prompt_len):
        labels[i] = -100 # Mask prompt loss

    return {"input_ids": input_ids, "attention_mask": tokenized_full["attention_mask"], "labels": labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # 1. Stability-First LoRA Config
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True # Protects against rank-induced instability
    )
    model = get_peft_model(model, lora_config)

    safe_data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            safe_data.append({"messages": row["messages"]})
            
    raw_dataset = Dataset.from_list(safe_data)
    dataset_split = raw_dataset.train_test_split(test_size=0.05, seed=42)
    
    def process_func(ex): return tokenize_conversation(tokenizer, ex["messages"], 2048)

    train_ds = dataset_split["train"].map(process_func, remove_columns=raw_dataset.column_names)
    eval_ds = dataset_split["test"].map(process_func, remove_columns=raw_dataset.column_names)

    # 3. Defensive Training Args
    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16, # Effective Batch Size: 32
        learning_rate=2e-5, # Slightly lower for stability
        num_train_epochs=3,
        bf16=True,
        logging_steps=5,
        eval_strategy="steps", # Watch for overfitting every 20 steps
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        load_best_model_at_end=True, # Automatically pick the least-overfit version
        metric_for_best_model="loss",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.1, # Regularization to prevent "forgetting"
        max_grad_norm=0.3,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    print("Started training...")
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Model have been saved to {args.out}")

if __name__ == "__main__": main()
#!/usr/bin/env python3
"""
train_grpo_qwen3_kgain.py

RL fine-tuning (GRPO) of a generator to maximize a trained reward model.

Training data (JSONL) should contain:
  {"paper_abstract": "...", "qa_annotations":[...]}

Key idea:
- Generator sees ONLY the abstract (prevents "writing to the test").
- Reward model sees (abstract + question set + generated article) to score Knowledge Gain.

Example (real run, LoRA, 8B generator on cuda:1, 4B RM on cuda:0):
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 train_grpo_qwen3_kgain.py \
    --data splits/rl_records_train.jsonl \
    --gen_model Qwen/Qwen3-8B \
    --gen_device cuda:1 \
    --bf16 --use_lora \
    --rm_model rm_qwen3_kgain \
    --rm_device cuda:0 \
    --rm_microbatch 1 \
    --out qwen3_8b_kgain_grpo_real \
    --num_generations 2 \
    --beta 0.04 \
    --max_new_tokens 900 \
    --max_rm_length 2048 \
    --steps 3000 \
    --grad_accum 8 \
    --per_device_batch_size 1 \
    --sample_every 100 \
    --sample_out samples_real.jsonl \
    --sample_max_new_tokens 900
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List

import torch

# TRL/PyTorch FSDP compatibility shim.
# Some torch builds do not export FSDPModule; TRL imports it during GRPO setup.
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
    # Generator sees ONLY the abstract (prevents “writing to the test”).
    # We also set an explicit target length and ask the model to stop when complete.
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
    # Reward model sees abstract + questions (defines the reward target).
    return (
        "You are scoring a candidate science news article.\n"
        "Score higher if it would increase a typical reader's ability to answer the questions after reading.\n\n"
        f"PAPER ABSTRACT:\n{abstract.strip()}\n\n"
        f"QUESTION SET:\n{qas_to_text(qas)}\n\n"
        "CANDIDATE CONTENT:\n"
    )


@torch.no_grad()
def make_rm_reward_func(
    rm_model,
    rm_tokenizer,
    device: torch.device,
    max_length: int,
    microbatch: int = 1,
):
    rm_model.eval()

    def reward_func(completions, rm_prompt, **kwargs):
        texts = [p + c for p, c in zip(rm_prompt, completions)]
        mb = max(1, int(microbatch))
        scores: List[float] = []

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


class SampleGenerationCallback(TrainerCallback):
    """
    Generates and logs a single sample completion every N steps.
    This is the easiest way to verify training is improving outputs and not reward-hacking.
    """

    def __init__(
        self,
        tokenizer,
        sample_prompt: str,
        out_path: str,
        every_steps: int,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.tokenizer = tokenizer
        self.sample_prompt = sample_prompt
        self.out_path = out_path
        self.every = max(0, int(every_steps))
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def on_step_end(self, args, state, control, **kwargs):
        if self.every <= 0:
            return control
        if state.global_step == 0 or (state.global_step % self.every) != 0:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        try:
            model_was_cache = getattr(model.config, "use_cache", None)
            model.config.use_cache = True

            enc = self.tokenizer(self.sample_prompt, return_tensors="pt", truncation=True)
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
            word_count = len(completion.split())

            row = {
                "step": int(state.global_step),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "words": word_count,
                "completion": completion,
            }
            with open(self.out_path, "a", encoding="utf-8") as w:
                w.write(json.dumps(row, ensure_ascii=False) + "\n")

            preview = completion[:900].replace("\n", " ")
            print(f"\n[SAMPLE @ step {state.global_step}] words={word_count} :: {preview}...\n")

            if model_was_cache is not None:
                model.config.use_cache = model_was_cache

        except Exception as e:
            print(f"[SAMPLE CALLBACK ERROR @ step {state.global_step}] {repr(e)}")

        return control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="rl_records.jsonl")
    ap.add_argument("--gen_model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--rm_model", required=True, help="path or HF id of trained RM")
    ap.add_argument("--out", default="qwen3_kgain_grpo")

    ap.add_argument("--max_prompt_length", type=int, default=1024, help="Soft cap applied by truncating prompt tokens")
    ap.add_argument("--max_rm_length", type=int, default=2048, help="RM input length cap (rm_prompt + completion)")
    ap.add_argument("--max_new_tokens", type=int, default=900, help="Max generated tokens for the news article")

    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--beta", type=float, default=0.04, help="KL weight; keep non-zero to avoid reward hacking")
    ap.add_argument("--num_generations", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--rm_device", default="cuda:0", help="Where to run reward model: cuda:0, cuda:1, or cpu")
    ap.add_argument("--rm_microbatch", type=int, default=1, help="Microbatch for RM scoring to avoid OOM")

    ap.add_argument("--gen_device", default="cuda:1", help="Device for generator, e.g. cuda:1")
    ap.add_argument("--bf16", action="store_true", help="Use bf16 for generator if available")
    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA for generator")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--sample_every", type=int, default=100, help="Generate and log 1 sample every N steps (0 disables)")
    ap.add_argument("--sample_out", default="samples_real.jsonl", help="Where to append sample generations")
    ap.add_argument("--sample_temp", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)
    ap.add_argument("--sample_max_new_tokens", type=int, default=900)

    args = ap.parse_args()

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
        rm_model,
        rm_tokenizer,
        rm_device,
        max_length=args.max_rm_length,
        microbatch=args.rm_microbatch,
    )

    gen_device = torch.device(args.gen_device)
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    device_map = None
    if gen_device.type == "cuda":
        device_map = {"": gen_device.index}

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

    if args.sample_every and args.sample_every > 0:
        trainer.add_callback(
            SampleGenerationCallback(
                tokenizer=gen_tok,
                sample_prompt=ds[0]["prompt"],
                out_path=args.sample_out,
                every_steps=args.sample_every,
                max_new_tokens=args.sample_max_new_tokens,
                temperature=args.sample_temp,
                top_p=args.sample_top_p,
            )
        )

    trainer.train()
    trainer.save_model(args.out)
    print(f"Saved GRPO-tuned generator to: {args.out}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
train_grpo_qwen3_kgain.py

GRPO fine-tuning of a generator to maximize a trained reward model.

Fixes:
- Prevent accidental resume / mixed checkpoints by backing up output_dir on fresh runs
- Truncate sample_out on fresh runs (no more mixed JSONL)
- Sampling callback uses model.eval() and do_sample=True
- More sophisticated generation prompt (headline + paragraph guide)
- Optional: sample_k to log multiple drafts per sampling step
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import re
import string
import os
import shutil
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List

import torch

# TRL/PyTorch FSDP compatibility shim
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


def build_generator_user_content(abstract: str) -> str:
    # More sophisticated prompt — but output stays as a single article (not JSON),
    MIN_WORDS = 450
    MAX_WORDS = 750
    PARAGRAPH_GUIDE = "12–20 short paragraphs (often 1–3 sentences each), separated by blank lines"
    HEADLINE_GUIDE = "Start with a punchy headline on the first line (no period)."

    return (
        f"ABSTRACT:\n{abstract.strip()}\n\n"
        "Write ONE science news article draft for a general audience.\n"
        "Output ONLY the article text (no JSON, no bullets, no outline, no preamble).\n\n"
        f"Style constraints:\n"
        f"- {HEADLINE_GUIDE}\n"
        f"- Length: {MIN_WORDS}–{MAX_WORDS} words\n"
        f"- Structure: {PARAGRAPH_GUIDE}\n"
        f"- Tone: explanatory, confident-but-not-hype, with concrete takeaways\n\n"
        "Content requirements:\n"
        "- Lead with the main finding in plain language.\n"
        "- Include 1–3 brief context paragraphs explaining why it matters.\n"
        "- Include key quantitative results ONLY if present in the abstract; otherwise stay qualitative.\n"
        "- Avoid deep protocol minutiae (no optimizer names, no measurement internals, etc.).\n"
        "- It is OK to say findings come from a study/paper.\n"
        "- Do NOT mention 'the abstract' or that you were given an abstract.\n"
        "- Do NOT invent facts: no made-up numbers, cohorts, years, institutions, author names, or journal names.\n"
        "- Only include a journal name if it is explicitly present in the abstract text.\n"
        "\n"
        "Stop when the article is complete.\n"
    )


def build_rm_prompt(abstract: str, qas: List[Dict[str, Any]]) -> str:
    return (
        "You are scoring a candidate science news article.\n"
        "Score higher if it would increase a typical reader's ability to answer the questions after reading.\n\n"
        f"PAPER ABSTRACT:\n{abstract.strip()}\n\n"
        f"QUESTION SET:\n{qas_to_text(qas)}\n\n"
        "CANDIDATE CONTENT:\n"
    )


def _text_stats(text: str) -> Dict[str, float]:
    t = text or ""
    chars = [c for c in t if not c.isspace()]
    n = len(chars)
    if n == 0:
        return {
            "alpha_ratio": 0.0,
            "digit_ratio": 0.0,
            "punct_ratio": 1.0,
            "word_count": 0,
            "word_alpha_ratio": 0.0,
        }

    alpha = sum(c.isalpha() for c in chars)
    digit = sum(c.isdigit() for c in chars)
    punct = sum(c in string.punctuation for c in chars)

    words = t.split()
    wc = len(words)
    good_words = sum(sum(ch.isalpha() for ch in w) >= 2 for w in words) if wc else 0

    return {
        "alpha_ratio": alpha / n,
        "digit_ratio": digit / n,
        "punct_ratio": punct / n,
        "word_count": wc,
        "word_alpha_ratio": (good_words / wc) if wc else 0.0,
    }


def _length_shape(words: int, min_words: int, max_words: int, weight: float) -> float:
    if words < min_words:
        return -weight * (min_words - words)
    if words > max_words:
        return -weight * (words - max_words)
    return 0.0


@torch.no_grad()
def make_rm_reward_func(
    rm_model,
    rm_tokenizer,
    device: torch.device,
    max_length: int,
    microbatch: int = 1,
    min_words: int = 420,
    max_words: int = 900,
    min_alpha_ratio: float = 0.72,
    min_word_alpha_ratio: float = 0.85,
    max_punct_ratio: float = 0.18,
    max_digit_ratio: float = 0.08,
    bad_reward: float = -200.0,
    len_weight: float = 0.02,
):
    rm_model.eval()
    bad_patterns = [
        r"\[math", r"\\frac", r"\\times", r"00000", r"\)\)\)", r"\]\]", r"\(\(", r"\)\)",
    ]
    bad_re = re.compile("|".join(bad_patterns), flags=re.IGNORECASE)

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

        final_rewards: List[float] = []
        for s, c in zip(rm_scores, completions):
            comp = (c or "").strip()
            st = _text_stats(comp)
            wc = int(st["word_count"])

            if (
                wc < min_words
                or wc > max_words
                or st["alpha_ratio"] < min_alpha_ratio
                or st["word_alpha_ratio"] < min_word_alpha_ratio
                or st["punct_ratio"] > max_punct_ratio
                or st["digit_ratio"] > max_digit_ratio
                or bad_re.search(comp) is not None
                or comp.lower().startswith("okay")
            ):
                final_rewards.append(float(bad_reward))
                continue

            r_len = _length_shape(wc, min_words=min_words, max_words=max_words, weight=len_weight)
            final_rewards.append(float(s) + float(r_len))

        return final_rewards

    return reward_func


class SampleGenerationCallback(TrainerCallback):
    """
    Generates and logs samples every N steps.

    Fixes:
    - model.eval() during generate
    - do_sample=True so temp/top_p/top_k apply
    - optional sample_k to get multiple drafts per step (logged as separate rows)
    """

    def __init__(
        self,
        tokenizer,
        sample_record: Dict[str, Any],
        out_path: str,
        every_steps: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        prompt_max_length: int,
        sample_k: int = 1,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 4,
        run_id: str = "",
    ):
        self.tokenizer = tokenizer
        self.sample_record = sample_record
        self.out_path = out_path
        self.every = max(0, int(every_steps))
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.prompt_max_length = int(prompt_max_length)
        self.sample_k = max(1, int(sample_k))
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.run_id = run_id or ""

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

            was_training = model.training
            model.eval()

            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.prompt_max_length,
            )
            device = next(model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            input_len = enc["input_ids"].shape[-1]

            with torch.inference_mode():
                out_ids = model.generate(
                    **enc,
                    do_sample=True,
                    num_return_sequences=self.sample_k,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # out_ids: [sample_k, seq_len] (since batch=1)
            if out_ids.dim() == 1:
                out_ids = out_ids.unsqueeze(0)

            for j in range(out_ids.shape[0]):
                completion_ids = out_ids[j, input_len:]
                completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                word_count = len(completion.split())

                row = {
                    "run_id": self.run_id,
                    "pid": os.getpid(),
                    "step": int(state.global_step),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sample_idx": int(j),
                    "article_id": article_id,
                    "words": word_count,
                    "paper_abstract": paper_abs,
                    "completion": completion,
                }
                with open(self.out_path, "a", encoding="utf-8") as w:
                    w.write(json.dumps(row, ensure_ascii=False) + "\n")

            if was_training:
                model.train()

        except Exception as e:
            with open(self.out_path, "a", encoding="utf-8") as w:
                w.write(json.dumps({"run_id": self.run_id, "step": int(state.global_step), "error": repr(e)}, ensure_ascii=False) + "\n")

        return control


def format_bonus(completions, **kwargs):
    out = []
    for c in completions:
        t = (c or "").strip()
        words = t.split()
        wc = len(words)

        chars = [ch for ch in t if not ch.isspace()]
        if not chars:
            out.append(-30.0)
            continue

        alpha = sum(ch.isalpha() for ch in chars) / len(chars)
        punct = sum(ch in string.punctuation for ch in chars) / len(chars)

        if t.startswith("A))))") or t.lower().startswith("okay") or "000000" in t or "\\frac" in t or "[math" in t:
            out.append(-50.0)
            continue

        if 450 <= wc <= 750 and alpha >= 0.70 and punct <= 0.18:
            out.append(+30.0)
        elif 250 <= wc <= 1000 and alpha >= 0.65 and punct <= 0.25:
            out.append(+5.0)
        else:
            out.append(-20.0)
    return out


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def backup_path_if_exists(path: str, label: str) -> None:
    if not path:
        return
    if os.path.exists(path):
        ts = time.strftime("%Y%m%d_%H%M%S")
        bak = f"{path}.bak_{ts}"
        shutil.move(path, bak)
        print(f"[fresh run] moved existing {label} {path} -> {bak}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="rl_records.jsonl")
    ap.add_argument("--gen_model", default="Qwen/Qwen3-8B")
    ap.add_argument("--rm_model", required=True, help="path or HF id of trained RM")
    ap.add_argument("--out", default="qwen3_kgain_grpo")

    ap.add_argument("--max_prompt_length", type=int, default=2048, help="Cap on generator prompt tokens")
    ap.add_argument("--max_rm_length", type=int, default=2048, help="RM input length cap")
    ap.add_argument("--max_new_tokens", type=int, default=1100, help="Generation budget for 450–750 words")

    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--beta", type=float, default=0.10, help="Stronger KL helps prevent RL collapse on small data")
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
    ap.add_argument("--save_total_limit", type=int, default=5)
    ap.add_argument("--resume_from_checkpoint", default=None)

    ap.add_argument("--log_file", default=None, help="Redirect stdout/stderr to this file")

    ap.add_argument("--sample_every", type=int, default=100)
    ap.add_argument("--sample_out", default="samples_real.jsonl")
    ap.add_argument("--sample_max_new_tokens", type=int, default=1100)
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.8)
    ap.add_argument("--sample_top_k", type=int, default=20)
    ap.add_argument("--sample_k", type=int, default=1, help="How many sampled drafts to log per sampling step")

    args = ap.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"

    log_fh = None
    if args.log_file:
        # overwrite log file to avoid mixing runs
        log_fh = open(args.log_file, "w", encoding="utf-8")
        try:
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass

    ctx_stdout = redirect_stdout(log_fh) if log_fh else nullcontext()
    ctx_stderr = redirect_stderr(log_fh) if log_fh else nullcontext()

    with ctx_stdout, ctx_stderr:
        print(f"[run_id] {run_id}", flush=True)
        print(f"[out_dir] {args.out}", flush=True)
        print(f"[sample_out] {args.sample_out}", flush=True)

        # If not resuming, prevent accidental resume/mixing by backing up existing output dir & sample file
        if args.resume_from_checkpoint in (None, "", "None"):
            backup_path_if_exists(args.out, "output_dir")
            if args.sample_out:
                backup_path_if_exists(args.sample_out, "sample_out")

        # Start a fresh sample_out with a header line (always append afterwards)
        if args.sample_out:
            with open(args.sample_out, "a", encoding="utf-8") as w:
                w.write(json.dumps({"run_id": run_id, "event": "start", "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False) + "\n")

        ds = load_dataset("json", data_files=args.data, split="train")

        gen_tok = AutoTokenizer.from_pretrained(args.gen_model, use_fast=True, trust_remote_code=True)
        if gen_tok.pad_token is None:
            gen_tok.pad_token = gen_tok.eos_token
        gen_tok.padding_side = "left"

        def make_gen_prompt_text(abstract: str) -> str:
            system = (
                "You are a science news writer.\n"
                "Write a clear, engaging news article for a general audience.\n"
                "It is OK to mention that findings come from a study/paper.\n"
                "Do NOT mention 'the abstract' or that you were given an abstract.\n"
                "Do NOT invent facts: no made-up numbers, cohorts, years, institutions, author names, or journal names.\n"
                "Only include a journal name if it is explicitly present in the abstract text.\n"
                "Prefer short paragraphs and a journalistic tone.\n"
                "Output ONLY the article text.\n"
            )
            user = build_generator_user_content(abstract)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            text = gen_tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return text

        def truncate_prompt(text: str) -> str:
            ids = gen_tok(text, add_special_tokens=False).input_ids
            if len(ids) <= args.max_prompt_length:
                return text
            ids = ids[: args.max_prompt_length]
            return gen_tok.decode(ids, skip_special_tokens=False)

        def _map(ex):
            abstract = ex["paper_abstract"]
            qas = ex["qa_annotations"]
            ex["prompt"] = truncate_prompt(make_gen_prompt_text(abstract))
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

            # keep rollout decoding controlled here
            generation_kwargs={
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 4,
                "eos_token_id": gen_tok.eos_token_id,
                "pad_token_id": gen_tok.pad_token_id,
            },

            log_completions=True,
            num_completions_to_print=2,

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
            reward_funcs=[reward_func, format_bonus],
        )

        if args.sample_every and args.sample_every > 0 and args.sample_out:
            trainer.add_callback(
                SampleGenerationCallback(
                    tokenizer=gen_tok,
                    sample_record=ds[0],
                    out_path=args.sample_out,
                    every_steps=args.sample_every,
                    max_new_tokens=args.sample_max_new_tokens,
                    temperature=args.sample_temperature,
                    top_p=args.sample_top_p,
                    top_k=args.sample_top_k,
                    prompt_max_length=args.max_prompt_length,
                    sample_k=args.sample_k,
                    repetition_penalty=1.05,
                    no_repeat_ngram_size=4,
                    run_id=run_id,
                )
            )

        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(args.out)
        print(f"Saved GRPO-tuned generator to: {args.out}", flush=True)

    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()
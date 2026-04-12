#!/usr/bin/env python3
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from tqdm import tqdm

# =========================
# CONFIG
# =========================
MODEL_NAME = "qwen3_8b_news_sft_strongteacher"
QA_MODEL_NAME = "Qwen/Qwen3-8B"
DATA_PATH = "qa_dataset.jsonl"

OUTPUT_DIR = "kg_rl_model"

MAX_NEW_TOKENS = 512
BATCH_SIZE = 2

LR = 3e-6

# =========================
# LOAD MODELS
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME, trust_remote_code=True)
qa_model = AutoModelForCausalLM.from_pretrained(
    QA_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.train()
qa_model.eval()

# =========================
# PROMPT
# =========================
def build_prompt(abstract):
    return f"""You are a science journalist.

Write a clear, accurate, and engaging science news article.

ABSTRACT:
{abstract}

Constraints:
- Do NOT invent facts
- Explain clearly for general audience
- Include key findings and implications

Output ONLY the article.
"""

# =========================
# GENERATION
# =========================
def generate_batch(prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# =========================
# QA REWARD (BATCHED)
# =========================
def qa_reward_batch(articles, batch_examples):
    rewards = []

    for article, ex in zip(articles, batch_examples):
        correct = 0
        questions = ex["questions"]

        for q in questions:
            prompt = f"""
ARTICLE:
{article}

QUESTION:
{q['question']}

Choices:
A. {q['choices'][0]}
B. {q['choices'][1]}
C. {q['choices'][2]}
D. {q['choices'][3]}

Answer with A/B/C/D only.
"""

            inputs = qa_tokenizer(prompt, return_tensors="pt").to(qa_model.device)

            out = qa_model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )

            pred = qa_tokenizer.decode(out[0], skip_special_tokens=True).strip()

            if pred.startswith(q["answer"]):
                correct += 1

        rewards.append(correct / len(questions))

    return rewards

# =========================
# PLACEHOLDERS (REPLACE WITH YOUR REAL FUNCTIONS)
# =========================
def grounding_score(article, abstract):
    return 0.0  # TODO: plug your real grounding

def accuracy_score(article, abstract):
    return 0.0  # TODO: plug your judge

# =========================
# TOTAL REWARD
# =========================
def compute_rewards(batch_examples, articles):
    qa_scores = qa_reward_batch(articles, batch_examples)

    rewards = []
    for ex, article, qa in zip(batch_examples, articles, qa_scores):

        grounding = grounding_score(article, ex["paper_abstract"])
        acc = accuracy_score(article, ex["paper_abstract"])

        reward = (
            0.65 * qa +
            0.20 * grounding +
            0.15 * acc
        )

        rewards.append(reward)

    # normalize
    rewards = np.array(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    # clip
    rewards = np.clip(rewards, -1, 1)

    return rewards.tolist()

# =========================
# LOAD DATA
# =========================
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.shuffle(seed=42)

# =========================
# PPO CONFIG
# =========================
ppo_config = PPOConfig(
    batch_size=BATCH_SIZE,
    mini_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=LR,
    log_with=None,
)

ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=ppo_config,
)

# =========================
# TRAIN LOOP
# =========================
for epoch in range(1):
    print(f"Epoch {epoch}")

    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset[i:i+BATCH_SIZE]

        prompts = [build_prompt(ex["paper_abstract"]) for ex in batch]

        query_tensors = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(model.device)

        # generate
        response_tensors = ppo_trainer.generate(
            query_tensors,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # rewards
        rewards = compute_rewards(batch, responses)

        # PPO step
        stats = ppo_trainer.step(
            list(query_tensors),
            list(response_tensors),
            rewards,
        )

        if i % 20 == 0:
            print(f"step={i} rewards={rewards}")

        if i % 200 == 0:
            save_path = os.path.join(OUTPUT_DIR, f"checkpoint_{i}")
            ppo_trainer.save_pretrained(save_path)

# final save
ppo_trainer.save_pretrained(OUTPUT_DIR)
print("Training complete.")
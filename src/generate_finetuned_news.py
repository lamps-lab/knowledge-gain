import json
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

CHECKPOINTS_ROOT = "outputs/sft_kg_top50_s2"
#"qwen3_4b" 

CHECKPOINT_STEPS = [40,60,70] 

INPUT_FILE = "../evaluation/generated_questions.json"
OUTPUT_DIR = "../evaluation/sft_outputs"

DEVICE = "cuda:0" # Swapped to 0, adjust if you are using a specific GPU index
MAX_NEW_TOKENS = 2100
MAX_INPUT_LENGTH = 4048


NUMERIC_FIDELITY_GUIDELINES = """
- Report all numerical information exactly as presented in the abstract.
- Use the same numerical values, units, ranges, and formats as the source.
- Present quantitative results in their original reported form without introducing alternative expressions.
- Keep numerical statements in the same structure and context as they appear in the abstract.
- Ensure that every number in the article directly corresponds to a value explicitly stated in the abstract.
- Use the study’s reported statistics directly rather than re-expressing them in different forms.

- Do not restate any numerical value in a different form or unit; retain the original representation used in the abstract.
"""

def build_generator_user_content(abstract: str, qa_annotations: list) -> str:
    return f"""
Numeric Fidelity Guidelines:
{NUMERIC_FIDELITY_GUIDELINES}

ABSTRACT:
{abstract.strip()}

Write a clear, engaging science news article.

Output ONLY the article text.
"""

def generate_news(model, tok, abstract: str, qa_annotations: list) -> str:
    system = f"""You are an expert science journalist for a major news outlet. 
    Numeric Fidelity Guidelines:
    {NUMERIC_FIDELITY_GUIDELINES}"""
    
    user = build_generator_user_content(abstract, qa_annotations)

    #print("\n--- PROMPT PREVIEW ---")
    #print(system)
    #print(user[:300] + "\n...\n[Prompt Truncated for display]\n...\n" + user[-300:])
    #print("--- END OF PROMPT ---\n")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    gen = tok.decode(
        out[0, enc["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    return gen


def load_model_and_tokenizer(model_dir: Path, tokenizer_dir: Path):
    print(f"\nLoading model from: {model_dir}")
    print(f"Loading tokenizer from: {tokenizer_dir}")

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # AutoPeftModelForCausalLM automatically merges the LoRA adapter with the base model
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": int(DEVICE.split(":")[1])} if DEVICE.startswith("cuda:") else "auto",
    )

    model.eval()
    return model, tok


def run_checkpoint(checkpoint_step: int, checkpoint_dir: Path):
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Note: Trainer saves the tokenizer alongside the checkpoint
    model, tok = load_model_and_tokenizer(
        model_dir=checkpoint_dir,
        tokenizer_dir=checkpoint_dir, 
    )

    print(f"\n{'#' * 80}")
    print(f"Running inference for checkpoint-{checkpoint_step}")
    print(f"{'#' * 80}")

    for i, entry in enumerate(dataset):
        # Updated key names to match generated_questions.json
        article_id = entry.get("article_id", "Unknown ID")
        abstract = entry.get("paper_abstract", "")

        print(f"\n{'=' * 60}")
        print(
            f"Checkpoint {checkpoint_step} | Entry {i+1}/{len(dataset)} "
            f"| ID: {article_id}"
        )
        print(f"{'=' * 60}")

        if not abstract or "Manual extraction needed" in abstract:
            print("  Skipping: No valid abstract text found for this entry.")
            entry[f"news_ckpt_{checkpoint_step}"] = ""
            continue

        final_news = generate_news(model, tok, abstract)

        print("\n--- GENERATED ARTICLE ---\n")
        print(final_news)
        print(f"\nGENERATED WORDS: {len(final_news.split())}")
        print("\n" + "-" * 60)

        # Save generation to a uniquely named key to prevent overwrites
        entry[f"news_0"] = final_news

    output_path = Path(OUTPUT_DIR) / f"generated_news_{checkpoint_step}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\nSaved checkpoint-{checkpoint_step} results to {output_path}")

    del model
    del tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    checkpoint_paths = {
        step: Path(CHECKPOINTS_ROOT) / f"checkpoint-{step}"
        for step in CHECKPOINT_STEPS
    }

    for step, ckpt_path in checkpoint_paths.items():
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    for step in CHECKPOINT_STEPS:
        run_checkpoint(step, checkpoint_paths[step])

    print("\nDone. All checkpoint outputs have been generated.")


if __name__ == "__main__":
    main()
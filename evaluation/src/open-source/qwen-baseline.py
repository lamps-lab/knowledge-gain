import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Hardcoded settings
# =========================
BASE_REPO = "Qwen/Qwen3-8B"   # base model, not finetuned
INPUT_FILE = "../../twenty_samples.json"
OUTPUT_FILE = "../../generated_news_base.json"
DEVICE = "cuda:1"
MAX_NEW_TOKENS = 1200
MAX_INPUT_LENGTH = 2048


def generate_article(model, tokenizer, abstract: str, news_length: int) -> str:
    system = (
        "You are an expert science journalist. "
        "Write a clear, engaging science news article for a general audience. "
        "Output only the article text."
    )

    user = (
        f"Write a science news article of about {news_length} words based on this abstract.\n\n"
        f"{abstract}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,#True,
            #temperature=0.7,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return generated_text


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(BASE_REPO, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_REPO,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": int(DEVICE.split(":")[1])} if DEVICE.startswith("cuda:") else None,
    )

    if not DEVICE.startswith("cuda:"):
        model = model.to(DEVICE)

    model.eval()

    for i, entry in enumerate(dataset):
        article_id = entry.get("id", "Unknown ID")
        category = entry.get("category", "Unknown Category")
        abstract = entry.get("abstract", "")
        news = entry.get("news", "")
        news_length = len(news.split()) if news else 500

        print(f"\n{'=' * 60}")
        print(
            f"Processing Entry {i+1}/{len(dataset)} | "
            f"ID: {article_id} | Category: {category} | target={news_length} words"
        )
        print(f"{'=' * 60}")

        if not abstract or "Manual extraction needed" in abstract:
            print("Skipping: No valid abstract text found for this entry.")
            continue

        generated_article = generate_article(model, tokenizer, abstract, news_length)

        print("\n--- GENERATED ARTICLE ---\n")
        print(generated_article)
        print(f"\nGENERATED WORDS: {len(generated_article.split())}")
        print("\n" + "-" * 60)

        entry["generated_news"] = generated_article

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\nDone. Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

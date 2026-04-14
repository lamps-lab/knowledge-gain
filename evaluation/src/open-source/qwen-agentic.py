import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_REPO = "Qwen/Qwen3-8B"   # base model
INPUT_FILE = "../../twenty_samples.json"
OUTPUT_FILE = "../../agentic_qwen.json"
DEVICE = "cuda:1"
MAX_NEW_TOKENS = 1200
MAX_INPUT_LENGTH = 2048
NUM_ITERATIONS = 2


def generate_response(model, tokenizer, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
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
            do_sample=False,
            #temperature=temperature,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return response


def drafter_agent(model, tokenizer, abstract: str, news_length: int) -> str:
    """Agent 1: Translates the abstract into an accessible news draft."""
    system_prompt = (
        "You are an expert science journalist. "
        "Your job is to take complex academic abstracts and turn them into engaging, "
        "accessible news articles for the general public. "
        "Create a catchy headline, keep the tone informative and exciting, and make the "
        "science easy to understand without dumbing it down. "
        "Output only the article text."
    )

    user_prompt = (
        f"Please draft a news article in about {news_length} words based on this abstract:\n\n"
        f"{abstract}"
    )

    return generate_response(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
    )


def revision_agent(model, tokenizer, abstract: str, draft: str) -> str:
    """Agent 2: Reviews the draft against the abstract and improves it."""
    system_prompt = (
        "You are a strict but fair senior editor at a top science magazine. "
        "Your job is to review article drafts to ensure they are factually accurate based "
        "on the original abstract, check for readability, and improve the narrative flow. "
        "Do not introduce new facts, statistics, or claims outside of the abstract. "
        "Output only the final polished article text."
    )

    user_prompt = f"""ORIGINAL ABSTRACT:
{abstract}

INITIAL DRAFT:
{draft}

Please revise and polish the draft. Ensure no scientific inaccuracies were introduced,
improve the hook, and output ONLY the final polished article including the headline.
"""

    return generate_response(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
    )


def generate_news_workflow(model, tokenizer, abstract: str, news_length: int) -> str:
    """Main function orchestrating the agentic workflow."""
    final_article = ""

    for i in range(NUM_ITERATIONS):
        print(f"iteration {i+1}")
        print("  ✍️  Drafter Agent is analyzing the abstract and writing the initial draft...")
        initial_draft = drafter_agent(model, tokenizer, abstract, news_length)

        print("  🧐  Revision Agent is fact-checking and polishing the draft...")
        final_article = revision_agent(model, tokenizer, abstract, initial_draft)

    return final_article


def main():
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{INPUT_FILE}'. Please check the path.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: '{INPUT_FILE}' is not a valid JSON file.")
        return

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

        print(f"\n{'='*60}")
        print(f"Processing Entry {i+1}/{len(dataset)} | ID: {article_id} | Category: {category} | {news_length} words")
        print(f"{'='*60}")

        if not abstract or "Manual extraction needed" in abstract:
            print("  ⚠️  Skipping: No valid abstract text found for this entry.")
            continue

        final_news = generate_news_workflow(model, tokenizer, abstract, news_length)

        print("\n--- FINAL PUBLISHED ARTICLE ---\n")
        print(final_news)
        print("\n" + "-"*60)

        entry["news_2"] = final_news

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\n✅ All finished! Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

# save as gen_checkpoint_check.py
# example:
# python3 gen_checkpoint_check.py --ckpt qwen3_8b_kgain/checkpoint-200 --device cuda:1

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def build_generator_user_content(abstract: str) -> str:
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
        "- Only include a journal name if it is explicitly present in the abstract text.\n\n"
        "Stop when the article is complete.\n"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint dir, e.g. qwen3_8b_kgain/checkpoint-200")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=1100)
    args = ap.parse_args()

    abstract = """We present a novel candidate for cold dark matter consisting of condensed Cooper pairs in a theory of interacting fermions with broken chiral symmetry. Establishing the thermal history from the early radiation era to the present, the fermions are shown to behave like standard radiation at high temperatures, but then experience a critical era decaying faster than radiation, akin to freeze-out, which sets the relic abundance. Through a second-order phase transition, fermion-antifermion pairs condense and the system asymptotes toward zero temperature and pressure. By the present era, the nonrelativistic, massive condensate decays slightly faster than in the standard scenario–a unique prediction that may be tested by combined measurements of the cosmic microwave background and large scale structure. We also show that in the case of massive fermions, the phase transition is frustrated, and instead leaves a residual, long-lived source of dark energy."""

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

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": int(args.device.split(":")[1])} if args.device.startswith("cuda:") else None,
    )
    model.eval()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=1.05,
            no_repeat_ngram_size=4,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    gen = tok.decode(out[0, enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    print(gen)
    print("\nWORDS:", len(gen.split()))

if __name__ == "__main__":
    main()
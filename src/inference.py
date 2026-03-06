# save as gen_baseline_check.py and run: python3 gen_baseline_check.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-8B"
abstract = """We present a novel candidate for cold dark matter consisting of condensed Cooper pairs in a theory of interacting fermions with broken chiral symmetry..."""

prompt = (
    "You are a science news writer.\n"
    "Write a clear, engaging news article for a general audience.\n"
    "Length: 450–750 words.\n"
    "Stop when the article is complete. Do NOT add filler.\n\n"
    f"ABSTRACT:\n{abstract}\n\n"
    "Write the news article now.\n"
)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": 0},
)
model.eval()

enc = tok(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(
        **enc,
        do_sample=False,            # greedy, deterministic
        max_new_tokens=600,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

gen = tok.decode(out[0, enc["input_ids"].shape[-1]:], skip_special_tokens=True)
print(gen[:1500])
print("\nWORDS:", len(gen.split()))

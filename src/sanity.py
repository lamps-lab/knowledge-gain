#!/usr/bin/env python3
import json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RM_PATH = "rm_qwen3_kgain"
DATA = "splits/rm_pairs_test.jsonl"
MAX_LEN = 4096

tok = AutoTokenizer.from_pretrained(RM_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

rm = AutoModelForSequenceClassification.from_pretrained(
    RM_PATH, trust_remote_code=True, torch_dtype="auto"
)
rm.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rm.to(device)

@torch.no_grad()
def score(txt: str) -> float:
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True).to(device)
    return rm(**enc).logits.squeeze(-1).float().item()

rows = [json.loads(l) for l in open(DATA, "r", encoding="utf-8") if l.strip()]
wins = 0
margins = []
for r in rows:
    s_ch = score(r["prompt"] + r["chosen"])
    s_rj = score(r["prompt"] + r["rejected"])
    wins += int(s_ch > s_rj)
    margins.append(s_ch - s_rj)

n = len(rows)
print(f"Test pairs: {n}")
print(f"Win rate: {wins}/{n} = {wins/n:.3f}")
print(f"Mean margin: {sum(margins)/n:+.3f}")
print(f"Min/Max margin: {min(margins):+.3f} / {max(margins):+.3f}")

import json
from collections import Counter

INPUT_FILE = "llm_judge_pairwise_news0_vs_news1_claude_sonnet.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

winner_counts = Counter()
presented_counts = Counter()

for rec in data:
    winner_counts[rec.get("winner_key")] += 1
    po = rec.get("presented_order") or {}
    presented_counts[(po.get("article_a"), po.get("article_b"))] += 1

print("Winner counts:", winner_counts)
print("Presented order counts:", presented_counts)

# show a few examples
for rec in data[:5]:
    print("-" * 80)
    print("id:", rec.get("id"))
    print("presented_order:", rec.get("presented_order"))
    print("winner_label:", rec.get("winner_label"))
    print("winner_key:", rec.get("winner_key"))
    print("reason:", (rec.get("judgment") or {}).get("reason"))


TARGET_IDS = {29733, 29734, 29739, 29753, 29775}

for rec in data:
    if rec.get("id") in TARGET_IDS:
        print("=" * 120)
        print("ID:", rec.get("id"))
        print("\nABSTRACT:\n", rec.get("abstract", "")[:2000])
        print("\nNEWS_0:\n", rec.get("news_0", "")[:2500])
        print("\nNEWS_1:\n", rec.get("news_1", "")[:2500])
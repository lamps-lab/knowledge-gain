import json

with open("pointwise.json", "r") as f:
    data = json.load(f)

results = []

for item in data:
    scores = item.get("scores_by_system", {})
    
    if "news_0" in scores and "news_2" in scores:
        kg_0 = scores["news_0"].get("knowledge_gain")
        kg_2 = scores["news_2"].get("knowledge_gain")
        
        if kg_0 is not None and kg_2 is not None:
            if kg_0 < 5 and kg_2 > kg_0:
                results.append({
                    "id": item["id"],
                    "kg_news_0": kg_0,
                    "kg_news_2": kg_2,
                    "reason_news_0": scores["news_0"].get("knowledge_gain_reason"),
                    "reason_news_2": scores["news_2"].get("knowledge_gain_reason"),
                    "accuracy_issue_news_2": scores["news_2"].get("accuracy_reason"),
                })

# pretty print
for r in results:
    print(f"\nID: {r['id']}")
    print(f"KG news_0: {r['kg_news_0']} | KG news_2: {r['kg_news_2']}")
    print("---- news_0 reason ----")
    print(r["reason_news_0"])
    print("---- news_2 reason ----")
    print(r["reason_news_2"])
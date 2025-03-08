import json
from sklearn.model_selection import train_test_split

# Load your articles dataset.
with open("../data/kgain_dataset.json", "r") as f:
    data = json.load(f)

# Flatten articles into individual QA samples.
flattened_samples = []
for article in data["articles"]:
    for ctype in ["abstract", "news", "tweet"]:
        content = article["contents"][ctype]
        for qa in article["qas"]:
            flattened_samples.append({
                "article_id": article["article_id"],  # Use the same DB article_id.
                "content_type": ctype,
                "content": content,
                "question": qa["question"],
                "answer": qa["answer"],
                "qa_type": qa["qa_type"]
            })

# Split based on article_id to avoid leakage.
article_ids = list({s["article_id"] for s in flattened_samples})
train_ids, test_ids = train_test_split(article_ids, test_size=0.3, random_state=42)
train_samples = [s for s in flattened_samples if s["article_id"] in train_ids]
test_samples  = [s for s in flattened_samples if s["article_id"] in test_ids]

# Save the splits.
with open("../data/train_dataset.json", "w") as f:
    json.dump(train_samples, f, indent=2)
with open("../data/test_dataset.json", "w") as f:
    json.dump(test_samples, f, indent=2)

print("Saved train_dataset.json and test_dataset.json")


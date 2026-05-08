import json
import math
import matplotlib.pyplot as plt
import numpy as np

INPUT_FILE = "results/open-source/pointwise_top50.json"
OUTPUT_FILE = "pointwise.pdf"

SYSTEMS = ["news_0", "news_1", "news_2", "news_3"]
SYSTEM_LABELS = {
    "news_0": "Finetuned",
    "news_1": "Baseline 1",
    "news_2": "Baseline 2",
    "news_3": "Baseline 3",
}

METRICS = ["accuracy", "completeness", "relevance", "clarity", "knowledge_gain"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "completeness": "Completeness",
    "relevance": "Relevance",
    "clarity": "Clarity",
    "knowledge_gain": "Knowledge Gain",
}


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def stddev(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

means = {system: [] for system in SYSTEMS}
stds = {system: [] for system in SYSTEMS}

for system in SYSTEMS:
    for metric in METRICS:
        vals = [
            rec["scores_by_system"][system][metric]
            for rec in data
            if system in rec["scores_by_system"] and metric in rec["scores_by_system"][system]
        ]
        means[system].append(mean(vals))
        stds[system].append(stddev(vals))

x = np.arange(len(METRICS))
width = 0.2

plt.figure(figsize=(12, 5))

for i, system in enumerate(SYSTEMS):
    plt.bar(
        x + (i - 1.5) * width,
        means[system],
        width=width,
        yerr=stds[system],
        capsize=4,
        label=SYSTEM_LABELS[system],
    )

plt.xticks(x, [METRIC_LABELS[m] for m in METRICS])
plt.ylim(1, 5.2)
plt.ylabel("Average Likert Score")
plt.xlabel("Metric")
#plt.title("Average LLM-as-a-Judge (Claude-Sonnet-4.6) Scores by Metric")
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=len(SYSTEMS),
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.savefig(OUTPUT_FILE, dpi=200)
plt.show()

print(f"Saved plot to {OUTPUT_FILE}\n")

print("Average scores ± std:")
for system in SYSTEMS:
    print(SYSTEM_LABELS[system])
    for metric, m, s in zip(METRICS, means[system], stds[system]):
        print(f"  {metric}: {m:.3f} ± {s:.3f}")

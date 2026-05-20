import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

POINTWISE_ARTICLE_IDS = [
    29733, 29734, 29739, 29753, 29775,
    29802, 29846, 29866, 29873, 29929,
    29902, 29943, 29958, 29988, 30052,
    30061, 30086, 30184, 30214, 30708,
]

DIMENSIONS = ["Accuracy", "Completeness", "Relevance", "Clarity"]

# 1. Load robust parsed human data
h_long = pd.read_csv("results/human_eval_analysis/pointwise_long.csv")

# Keep valid scores only
h_long = h_long.dropna(subset=["score"]).copy()

# 2. Plot human pointwise distributions
plt.figure(figsize=(10, 6))

melted_h = h_long.rename(columns={"dimension": "Metric", "score": "Score", "system": "Group"})

sns.boxplot(
    data=melted_h,
    x="Metric",
    y="Score",
    hue="Group",
    palette="Set2",
    showfliers=False,
)

sns.stripplot(
    data=melted_h,
    x="Metric",
    y="Score",
    hue="Group",
    dodge=True,
    color=".3",
    alpha=0.45,
    legend=False,
)

plt.ylabel("Human score (1--5, higher is better)", fontsize=14)
plt.xlabel("")
#plt.title("Human Pointwise Ratings: Ours vs Agentic", fontsize=16)
plt.ylim(0.5, 5.5)
plt.grid(axis="y", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig("human_pointwise.pdf")
plt.close()

# 3. Aggregate human ratings by article/system/dimension
h_agg = (
    h_long
    .groupby(["article_id", "system_key", "system", "dimension"], as_index=False)["score"]
    .mean()
)

h_wide = h_agg.pivot_table(
    index=["article_id", "system_key", "system"],
    columns="dimension",
    values="score",
).reset_index()

h_wide.columns.name = None

# 4. Load LLM judge data
with open("results/open-source/pointwise_top50.json", "r", encoding="utf-8") as f:
    llm_data = json.load(f)

llm_records = []

for story in llm_data:
    article_id = int(story["id"])

    for system_key, scores in story.get("scores_by_system", {}).items():
        if system_key not in {"news_0", "news_2"}:
            continue

        llm_records.append(
            {
                "article_id": article_id,
                "system_key": system_key,
                "Accuracy_LLM": scores.get("accuracy"),
                "Completeness_LLM": scores.get("completeness"),
                "Relevance_LLM": scores.get("relevance"),
                "Clarity_LLM": scores.get("clarity"),
                "ExpectedKGain_LLM": scores.get("knowledge_gain"),
                "MeanScore_LLM": scores.get("mean_score"),
            }
        )

l_df = pd.DataFrame(llm_records)

# 5. Join by article_id + system_key
merged = h_wide.merge(
    l_df,
    on=["article_id", "system_key"],
    how="inner",
)

merged["MeanScore_Human"] = merged[DIMENSIONS].mean(axis=1)

print("Merged rows:", len(merged))
print(merged[["article_id", "system_key", "system"]].head())

# 6. Overall human--LLM correlation across matched dimensions
human_vals = []
llm_vals = []
dim_labels = []

for dim in DIMENSIONS:
    h_col = dim
    l_col = f"{dim}_LLM"

    sub = merged[[h_col, l_col]].dropna()

    human_vals.extend(sub[h_col].tolist())
    llm_vals.extend(sub[l_col].tolist())
    dim_labels.extend([dim] * len(sub))

plot_df = pd.DataFrame(
    {
        "Human Mean Score": human_vals,
        "LLM Judge Score": llm_vals,
        "Dimension": dim_labels,
    }
)

pearson_r, pearson_p = pearsonr(plot_df["Human Mean Score"], plot_df["LLM Judge Score"])
spearman_r, spearman_p = spearmanr(plot_df["Human Mean Score"], plot_df["LLM Judge Score"])

print(f"Overall Pearson r = {pearson_r:.3f}, p = {pearson_p:.3f}")
print(f"Overall Spearman rho = {spearman_r:.3f}, p = {spearman_p:.3f}")

# 7. Pretty scatter plot
plt.figure(figsize=(7, 6))

sns.regplot(
    data=plot_df,
    x="Human Mean Score",
    y="LLM Judge Score",
    scatter=False,
    line_kws={"color": "red", "linestyle": "--"},
)

sns.scatterplot(
    data=plot_df,
    x="Human Mean Score",
    y="LLM Judge Score",
    hue="Dimension",
    s=70,
    alpha=0.75,
)

plt.xlabel("Human mean score", fontsize=14)
plt.ylabel("LLM judge score", fontsize=14)
#plt.title(f"Human--LLM Pointwise Alignment\nPearson r = {pearson_r:.2f}, Spearman $\\rho$ = {spearman_r:.2f}", fontsize=14)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("human_llm_pointwise_correlation.pdf")
plt.close()

# 8. Per-dimension correlations
rows = []

for dim in DIMENSIONS:
    h_col = dim
    l_col = f"{dim}_LLM"
    sub = merged[[h_col, l_col]].dropna()

    if len(sub) >= 2:
        pr, pp = pearsonr(sub[h_col], sub[l_col])
        sr, sp = spearmanr(sub[h_col], sub[l_col])
    else:
        pr, pp, sr, sp = np.nan, np.nan, np.nan, np.nan

    rows.append(
        {
            "Dimension": dim,
            "Pearson r": pr,
            "Pearson p": pp,
            "Spearman rho": sr,
            "Spearman p": sp,
            "n": len(sub),
        }
    )

corr_df = pd.DataFrame(rows)
print("\nPer-dimension correlations:")
print(corr_df.round(3))

corr_df.to_csv("human_llm_pointwise_correlations.csv", index=False)

wide = merged.pivot_table(
    index="article_id",
    columns="system",
    values=["MeanScore_Human", "MeanScore_LLM"],
    aggfunc="mean",
)

# Mean-score crossplot: one point per matched article/system item
mean_plot_df = merged[["MeanScore_Human", "MeanScore_LLM", "system"]].dropna().copy()

mean_pearson_r, mean_pearson_p = pearsonr(
    mean_plot_df["MeanScore_Human"],
    mean_plot_df["MeanScore_LLM"],
)
mean_spearman_r, mean_spearman_p = spearmanr(
    mean_plot_df["MeanScore_Human"],
    mean_plot_df["MeanScore_LLM"],
)

print(f"Mean-score Pearson r = {mean_pearson_r:.3f}, p = {mean_pearson_p:.3f}")
print(f"Mean-score Spearman rho = {mean_spearman_r:.3f}, p = {mean_spearman_p:.3f}")

plt.figure(figsize=(7, 6))

sns.regplot(
    data=mean_plot_df,
    x="MeanScore_Human",
    y="MeanScore_LLM",
    scatter=False,
    line_kws={"color": "red", "linestyle": "--"},
)

sns.scatterplot(
    data=mean_plot_df,
    x="MeanScore_Human",
    y="MeanScore_LLM",
    hue="system",
    s=80,
    alpha=0.8,
)

plt.xlabel("Human mean score", fontsize=14)
plt.ylabel("LLM judge mean score", fontsize=14)
#plt.title(
#    f"Human--LLM Mean Pointwise Alignment\n"
#    f"Pearson r = {mean_pearson_r:.2f}, Spearman $\\rho$ = {mean_spearman_r:.2f}",
#    fontsize=14,
#)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("human_llm_pointwise_mean_correlation_2.pdf")
plt.close()
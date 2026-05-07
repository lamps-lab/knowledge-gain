import pandas as pd
from scipy import stats

df = pd.read_csv("analysis_merged_metric_points.csv")

# System-level means
print(df.groupby("article_key")[[
    "human_pre_acc",
    "human_post_acc",
    "human_kgain",
    "human_norm_kgain",
    "llmsim_pre_acc",
    "llmsim_post_acc",
    "llmsim_kgain",
    "llmsim_norm_kgain",
]].mean())

# Within-topic deltas: Ours news_0 minus baseline news_2
wide = df.pivot(index="article_id", columns="article_key", values=[
    "human_norm_kgain",
    "llmsim_norm_kgain",
    "human_kgain",
    "llmsim_kgain",
])

wide["delta_human_norm"] = wide[("human_norm_kgain", "news_0")] - wide[("human_norm_kgain", "news_2")]
wide["delta_llmsim_norm"] = wide[("llmsim_norm_kgain", "news_0")] - wide[("llmsim_norm_kgain", "news_2")]

sub = wide[["delta_human_norm", "delta_llmsim_norm"]].dropna()

print("n =", len(sub))
print("Spearman:", stats.spearmanr(sub["delta_llmsim_norm"], sub["delta_human_norm"]))
print("Kendall:", stats.kendalltau(sub["delta_llmsim_norm"], sub["delta_human_norm"]))

sign_agree = (
    (sub["delta_llmsim_norm"] > 0) == (sub["delta_human_norm"] > 0)
).mean()
print("Sign agreement:", sign_agree)

print(sub.describe())
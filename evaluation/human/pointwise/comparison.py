import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 1. Process Human Data
df = pd.read_csv('pointwise-internal-human.csv', low_memory=False)
def recode(v):
    try: return 6 - float(v)
    except: return np.nan

# Extract human scores into long-form
human_list = []
dimensions = ["Accuracy", "Completeness", "Relevance", "Clarity"]
for i in range(20): # For each article
    start_idx = 17 + 8 * i
    for r_idx in range(2, len(df)): # For each student
        ratings = [recode(df.iloc[r_idx, start_idx + j*2]) for j in range(4)]
        human_list.append({"Article_ID": i+1, "Accuracy": ratings[0], 
                            "Completeness": ratings[1], "Relevance": ratings[2], "Clarity": ratings[3]})

h_df = pd.DataFrame(human_list)

# 2. Parse LLM Judge JSON
llm_json = [...] # Paste your JSON here
llm_scores = []
for sys, s in llm_json[0]['scores_by_system'].items():
    llm_scores.append({"System": sys, "Accuracy": s['accuracy'], 
                       "Completeness": s['completeness'], "Relevance": s['relevance'], "Clarity": s['clarity']})
l_df = pd.DataFrame(llm_scores)

# 3. Calculate Correlation (Human Mean vs LLM)
h_means = h_df[h_df['Article_ID'] <= 4].groupby('Article_ID')[dimensions].mean().values.flatten()
l_vals = l_df.sort_values('System')[dimensions].values.flatten()
corr, _ = pearsonr(h_means, l_vals)

# 4. Create the Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Distribution of 5 students (Box + Points)
sns.boxplot(data=h_df.melt(value_vars=dimensions), x='variable', y='value', ax=ax1, palette="Set3")
sns.stripplot(data=h_df.melt(value_vars=dimensions), x='variable', y='value', ax=ax1, color=".2", alpha=0.5)
ax1.set_title("Participant Score Distribution (N=5)")

# Right: Human Mean vs LLM Judge Mean
x = np.arange(len(dimensions))
ax2.bar(x - 0.2, h_df[h_df['Article_ID'] <= 4][dimensions].mean(), 0.4, label='Human Mean')
ax2.bar(x + 0.2, l_df[dimensions].mean(), 0.4, label='LLM Judge')
ax2.set_xticks(x), ax2.set_xticklabels(dimensions)
ax2.set_title(f"Comparison (Correlation: {corr:.2f})")
ax2.legend()

plt.show()
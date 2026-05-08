import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('pointwise.csv', low_memory=False)

# 2. Configuration
# Indices for Finetuned and Agentic articles
finetuned_ids = [1, 4, 6, 7, 9, 12, 15, 16, 18, 19]
agentic_ids = [2, 3, 5, 8, 10, 11, 13, 14, 17, 20]
dimensions = ["Accuracy", "Completeness", "Relevance", "Clarity"]

# Recoding function: Option 1 -> Score 5, Option 5 -> Score 1
def recode_score(val):
    try:
        v = float(val)
        return 6 - v if not np.isnan(v) else np.nan
    except:
        return np.nan

# 3. Data Extraction Logic
def get_article_ratings(row, article_id):
    # Each article now takes 8 columns (Rating, Reasoning, Rating, Reasoning...)
    # Starting index is 17 (the first 'Accuracy' column)
    start_idx = 17 + 8 * (article_id - 1)
    
    # We only want the rating columns (relative indices 0, 2, 4, 6)
    # Skipping indices 1, 3, 5, 7 which contain the reasoning text
    rating_indices = [start_idx, start_idx + 2, start_idx + 4, start_idx + 6]
    raw_vals = row.iloc[rating_indices].values
    
    return [recode_score(v) for v in raw_vals]

# 4. Process all respondents (starting from index 2)
all_ft_data = []
all_ag_data = []

for i in range(2, len(df)):
    resp_row = df.iloc[i]
    
    # Extract ratings for this respondent
    ft_ratings = [get_article_ratings(resp_row, aid) for aid in finetuned_ids]
    ag_ratings = [get_article_ratings(resp_row, aid) for aid in agentic_ids]
    
    all_ft_data.extend(ft_ratings)
    all_ag_data.extend(ag_ratings)

# Create Master DataFrames
ft_df = pd.DataFrame(all_ft_data, columns=dimensions)
ag_df = pd.DataFrame(all_ag_data, columns=dimensions)

# 5. Output Statistics
print("--- Finetuned Mean Scores ---")
print(ft_df.mean())
print("\n--- Agentic Mean Scores ---")
print(ag_df.mean())

# 6. Visualization
x = np.arange(len(dimensions))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, ft_df.mean(), width, label='Finetuned', color='#4A90E2')
plt.bar(x + width/2, ag_df.mean(), width, label='Agentic', color='#F5A623')

plt.ylabel('Mean Score (1-5, Higher is Better)')
plt.title('Human Evaluation Comparison: Finetuned vs Agentic')
plt.xticks(x, dimensions)
plt.ylim(0, 5)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('human_evaluation_results.pdf')
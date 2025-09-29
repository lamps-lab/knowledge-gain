import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

mapping = {'A': 'News', 'B': 'Abstract', 'C': 'Tweet'}

survey_files = {
    "A": ["a1", # "a2",
          "a3"],
    "B": ["b1", "b2" # "b3",
         ],
    "C": [# "c1", 
          "c2", "c3"]
}

sc0_grouped, sc1_grouped, gain_grouped = [], [], []

# collect all survey data into pooled groups
for g, subs in survey_files.items():
    all_sc0, all_sc1, all_gain = [], [], []
    for sub in subs:
        df = pd.read_csv(Path(f'../data/newdata/{sub}.csv'), skiprows=[1])
        df[['SC0', 'SC1']] = df[['SC0', 'SC1']].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=['SC0', 'SC1'])

        all_sc0.extend(df['SC0'].values)
        all_sc1.extend(df['SC1'].values)
        all_gain.extend((df['SC1'] - df['SC0']).values)

    sc0_grouped.append(np.array(all_sc0))
    sc1_grouped.append(np.array(all_sc1))
    gain_grouped.append(np.array(all_gain))

all_scores = np.concatenate(sc0_grouped + sc1_grouped + gain_grouped)

# ── plotting ────────────────────────────────────────────────
x, width = range(len(mapping)), 0.25
fig, ax = plt.subplots(figsize=(9, 7))

for i, (p1, p2, gain) in enumerate(zip(sc0_grouped, sc1_grouped, gain_grouped)):
    # Part 1
    ax.boxplot(p1, positions=[i - width], widths=width*0.9, patch_artist=True,
               showfliers=False, boxprops=dict(facecolor='C0'),
               medianprops=dict(color='orange'))

    # Part 2
    ax.boxplot(p2, positions=[i], widths=width*0.9, patch_artist=True,
               showfliers=False, boxprops=dict(facecolor='C1'),
               medianprops=dict(color='blue', linewidth=1))

    # Gain
    ax.boxplot(gain, positions=[i + width], widths=width*0.9, patch_artist=True,
               showfliers=False, boxprops=dict(facecolor='C2'),
               medianprops=dict(color='orange'))

# axis dressing
ax.set_xticks(list(x))
ax.set_xticklabels(list(mapping.values()), fontsize=14)
ax.set_ylabel('Score distribution', fontsize=16)
ax.set_ylim(0, all_scores.max() * 1.25)

# legend
handles = [
    plt.Line2D([0], [0], marker='s', linestyle='', markerfacecolor='C0',
               markeredgecolor='none', markersize=15, label='Part 1'),
    plt.Line2D([0], [0], marker='s', linestyle='', markerfacecolor='C1',
               markeredgecolor='none', markersize=15, label='Part 2'),
    plt.Line2D([0], [0], marker='s', linestyle='', markerfacecolor='C2',
               markeredgecolor='none', markersize=15, label='Knowledge Gain')
]
ax.legend(handles=handles, frameon=False, fontsize=13)

plt.tight_layout()
plt.show()

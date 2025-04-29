import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

groups  = ['A', 'B', 'C']
mapping = {'A': 'News', 'B': 'Abstract', 'C': 'Tweet'}

sc0_vals_all, sc1_vals_all, diff_vals_all = [], [], []

for g in groups:
    df = pd.read_csv(Path(f'./data/group{g}.csv'), skiprows=[1])
    df[['SC0', 'SC1']] = df[['SC0', 'SC1']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['SC0', 'SC1'])              # keep complete rows only

    sc0_vals_all.append(df['SC0'].values)
    sc1_vals_all.append(df['SC1'].values)
    diff_vals_all.append((df['SC1'] - df['SC0']).values)

all_scores = np.concatenate(sc0_vals_all + sc1_vals_all + diff_vals_all)

# ── plotting ────────────────────────────────────────────────────────────────
x, width = range(len(groups)), 0.25
fig, ax  = plt.subplots(figsize=(9, 7))

for i, (p1, p2, gain) in enumerate(zip(sc0_vals_all, sc1_vals_all, diff_vals_all)):
    # Part 1  (blue box, orange median)
    ax.boxplot(p1,
               positions=[i - width],
               widths=width*0.9,
               patch_artist=True,
               showfliers=False,
               boxprops=dict(facecolor='C0'),
               medianprops=dict(color='orange'))

    # Part 2  (orange box, **blue** median)
    ax.boxplot(p2,
               positions=[i],
               widths=width*0.9,
               patch_artist=True,
               showfliers=False,
               boxprops=dict(facecolor='C1'),
               medianprops=dict(color='blue', linewidth=1))

    # Knowledge-gain (green box, orange median)
    ax.boxplot(gain,
               positions=[i + width],
               widths=width*0.9,
               patch_artist=True,
               showfliers=False,
               boxprops=dict(facecolor='C2'),
               medianprops=dict(color='orange'))

# axis dressing
ax.set_xticks(list(x))
ax.set_xticklabels([mapping[g] for g in groups], fontsize=14)
ax.set_ylabel('Score distribution', fontsize=16)
ax.set_ylim(0, all_scores.max() * 1.25)

# legend (proxy handles match box/median colours)
handles = [
    plt.Line2D([0], [0], marker='s', linestyle='',
               markerfacecolor='C0', markeredgecolor='none', markersize=15,
               label='Part 1'),
    plt.Line2D([0], [0], marker='s', linestyle='',
               markerfacecolor='C1', markeredgecolor='none', markersize=15,
               label='Part 2'),
    plt.Line2D([0], [0], marker='s', linestyle='',
               markerfacecolor='C2', markeredgecolor='none', markersize=15,
               label='Knowledge Gain')
]
ax.legend(handles=handles, frameon=False, fontsize=13)

plt.tight_layout()
plt.show()

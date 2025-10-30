import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

groups = ['A', 'B', 'C']
plt.figure(figsize=(12, 7))

for g in groups:
	if g == "B":
		fp = Path(f'.oldkgain-survey/')
    fp = Path(f'./data/group{g}.csv')
    df = pd.read_csv(fp, skiprows=[1])          # skip Qualtrics mapping row

    # ── drop the mapping row if it slipped through ───────────────────────────
    if df.iloc[0].astype(str).str.contains("ImportId").any():
        df = df.iloc[1:].reset_index(drop=True)

    # ── pick the Timer_Page Submit columns and force them numeric ────────────
    timer_cols = [c for c in df.columns if "Timer_Page Submit" in c]
    timers = (df[timer_cols]
              .apply(pd.to_numeric, errors='coerce'))      # coerce → NaN, OK

    # ── aggregate across *rows* (participants) – use mean, median if preferred
    mean_times = timers.mean(skipna=True)

    # sample index = running question/block number (1 … n)
    x = range(1, len(mean_times) + 1)
    label = {"A": "news", "B": "abstract", "C": "tweet"}[g]

    # plot the per-question line and remember its colour
    line_obj, = plt.plot(x, mean_times, marker='o', label=label)
    colour = line_obj.get_color()

    # ── same-colour dashed mean line, no extra legend entry ────────────────
    plt.axhline(mean_times.mean(),
                linestyle='--',
                linewidth=1.2,
                color=colour,
                alpha=0.8)

plt.xlabel("Sample", fontsize=16)
plt.ylabel("Mean time (s)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, alpha=.4)
plt.legend(frameon=False, fontsize=16)
plt.tight_layout()
plt.show()

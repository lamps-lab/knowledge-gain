import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# block to survey mapping
block_map = {
    1: {"a": "a1", "b": "b1", "c": "c1"},   # docs 1–10
    2: {"a": "a2", "b": "b2", "c": "c2"},   # docs 11–20
    3: {"a": "a3", "b": "b3", "c": "c3"}    # docs 21–30
}

label_map = {"a": "News", "b": "Abstract", "c": "Tweet"}
colors    = {"a": "C0",   "b": "C1",       "c": "C2"}
offsets   = {"a": -0.25,  "b": 0.0,        "c": 0.25}   # per-document offsets

# cache survey files so we read each csv only once
survey_cache = {}

def get_timers(sub):
    """Return numeric Timer_Page Submit columns for a survey file (cached)."""
    if sub in survey_cache:
        return survey_cache[sub]
    fp = Path(f"../data/newdata/{sub}.csv")
    df = pd.read_csv(fp, skiprows=[1])
    # drop the Qualtrics mapping row if it slipped in
    if df.iloc[0].astype(str).str.contains("ImportId").any():
        df = df.iloc[1:].reset_index(drop=True)
    timer_cols = [c for c in df.columns if "Timer_Page Submit" in c]
    timers = df[timer_cols].apply(pd.to_numeric, errors="coerce")
    survey_cache[sub] = timers
    return timers

# collect per-document, per-group distributions
data_series, positions, box_colors = [], [], []

for doc in range(1, 31):
    block = (doc - 1) // 10 + 1        # 1..3
    rel   = (doc - 1) % 10             # 0..9 column inside the block

    for g in ("a", "b", "c"):          # order: News, Abstract, Tweet
        sub = block_map[block].get(g)
        if not sub:                    # missing survey -> no box for this group/doc
            continue
        timers = get_timers(sub)
        if rel >= timers.shape[1]:
            continue                   # this survey file has fewer than 10 docs
        s = timers.iloc[:, rel].dropna()
        if s.empty:
            continue

        data_series.append(s)
        positions.append(doc + offsets[g])
        box_colors.append(colors[g])

# plot grouped boxplots (three per document)
fig, ax = plt.subplots(figsize=(18, 7))

bp = ax.boxplot(
    data_series,
    positions=positions,
    widths=0.22,
    patch_artist=True,
    showfliers=True
)

# color each box and style medians
for box, color in zip(bp["boxes"], box_colors):
    box.set_facecolor(color)
for med in bp["medians"]:
    med.set_color("black")
    med.set_linewidth(1.2)

# axes & cosmetics
ax.set_xlim(0.5, 30.5)
ax.set_xticks(range(1, 31))             # positions
ax.set_xticklabels(range(1, 31))        # labels as integers
ax.set_xlabel("Document", fontsize=15)
ax.set_ylabel("Time distribution (s)", fontsize=15)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=12)
ax.grid(True, alpha=0.35)


# block separators
for x in (10.5, 20.5):
    ax.axvline(x, color="gray", linestyle="--", alpha=0.5)

# legend
handles = [plt.Line2D([0], [0], marker="s", linestyle="", markerfacecolor=colors[g],
                      markeredgecolor="none", markersize=12, label=label_map[g])
           for g in ("a", "b", "c")]
ax.legend(handles=handles, frameon=False, fontsize=12, loc="upper right")

plt.tight_layout()
plt.show()



summed_data = {g: [] for g in ("a", "b", "c")}

for block in block_map:
    for g in ("a", "b", "c"):
        sub = block_map[block].get(g)
        if not sub:
            continue
        timers = get_timers(sub)
        # flatten all timer values into one list (ignoring NaNs)
        vals = timers.to_numpy().flatten()
        vals = vals[~pd.isna(vals)]
        if len(vals) > 0:
            summed_data[g].extend(vals)

# create the boxplot data in label order
group_order = ["a", "b", "c"]
all_series = [summed_data[g] for g in group_order]
all_colors = [colors[g] for g in group_order]
all_labels = [label_map[g] for g in group_order]

fig, ax = plt.subplots(figsize=(6, 7))
bp2 = ax.boxplot(
    all_series,
    patch_artist=True,
    widths=0.5,
    showfliers=True
)

for box, color in zip(bp2["boxes"], all_colors):
    box.set_facecolor(color)
for med in bp2["medians"]:
    med.set_color("black")
    med.set_linewidth(1.3)

ax.set_xticks(range(1, len(all_labels) + 1))
ax.set_xticklabels(all_labels, fontsize=12)
ax.set_ylabel("Time distribution (s)", fontsize=14)
ax.set_title("Summed Distributions Across All Documents", fontsize=15)
ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.show()


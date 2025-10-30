#!/usr/bin/env python
# kgain_barplots_json.py
# --------------------------------------------------------------------------
# Draw clustered bar-plots of Part-1, Part-2 accuracy and Knowledge-gain
# for (a) content types and (b) question types.
# --------------------------------------------------------------------------

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
INFILE         = Path("./7b/baseline_results.json")   # adjust if needed
OUT_CONTENT_FIG = Path("kgain_by_content_type_bar.png")
OUT_QTYPE_FIG   = Path("kgain_by_question_type_bar.png")
# --------------------------------------------------------------------------


# ---------- load & enrich --------------------------------------------------
with open(INFILE) as f:
    recs = json.load(f)

df = pd.DataFrame(recs)
df.columns = [c.lower().strip() for c in df.columns]          # normalise

# fractional per-item accuracy & gain
df["acc_part1"] = df["points_part1"] / df["max_points"]
df["acc_part2"] = df["points_part2"] / df["max_points"]
df["kgain"]     = df["acc_part2"]   - df["acc_part1"]

# helper --------------------------------------------------------------------
def plot_group_bars(df, group_col, order=None, label_map=None,
                    title="", outfile=None):
    """
    Aggregate accuracy & Δ-accuracy by `group_col` and plot a 3-bar cluster
    (Part 1, Part 2, Gain) for each group.
    """
    if label_map is None:
        label_map = {g: g.capitalize() for g in df[group_col].unique()}

    # micro-accuracy aggregation
    agg = (df.groupby(group_col)
             [["points_part1", "points_part2", "max_points"]]
             .sum())

    if order is not None:            # enforce a specific left-to-right order
        agg = agg.reindex(order)

    acc1 = agg["points_part1"] / agg["max_points"]
    acc2 = agg["points_part2"] / agg["max_points"]
    gain = acc2                    - acc1

    # plotting --------------------------------------------------------------
    x      = np.arange(len(agg))
    width  = 0.25
    max_y  = max(acc1.max(), acc2.max(), gain.max())

    fig, ax = plt.subplots(figsize=(9, 7))

    b1 = ax.bar(x - width, acc1, width, label="Part 1", color="C0")
    b2 = ax.bar(x,         acc2, width, label="Part 2", color="C1")
    b3 = ax.bar(x + width, gain, width, label="Knowledge Gain", color="C2")

    #labels = [f"{patch.get_height()*100:.1f}%" for patch in bars]
    #ax.bar_label(bars, labels=labels, padding=3, fontsize=12)
    for bars in (b1, b2, b3):
        labels = [f"{patch.get_height()*100:.1f}%" for patch in bars]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels([label_map[g] for g in agg.index], fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_ylim(0, max_y * 1.25 if max_y > 0 else 1)
    ax.set_title(title, fontsize=16, pad=15)
    ax.legend(frameon=False, fontsize=14)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
        print(f"Figure saved → {outfile.resolve()}")
    plt.show()


# ---------- content-type plot ---------------------------------------------
content_order = ["news", "abstract", "tweet"]          # tweak if wanted
content_label = {"news":"News", "abstract":"Abstract", "tweet":"Tweet"}

plot_group_bars(df,
                group_col="content_type",
                order=content_order,
                label_map=content_label,
                #title="Knowledge gain by content type",
                outfile=OUT_CONTENT_FIG)


# ---------- question-type plot --------------------------------------------
# Define a nice order if you like, else let .unique() decide
qtype_order = ["tf", "mc_easy", "mc_hard"]
qtype_label = {"tf":"True / False",
               "mc_easy":"Easy MC",
               "mc_hard":"Hard MC"}

# Standardise qa_type strings (lower-case, underscores)
df["qa_type"] = (df["qa_type"]
                 .str.lower()
                 .str.replace(r"[^\w]+", "_", regex=True))

plot_group_bars(df,
                group_col="qa_type",
                order=[q for q in qtype_order if q in df["qa_type"].unique()],
                label_map=qtype_label,
                #title="Knowledge gain by question type",
                outfile=OUT_QTYPE_FIG)

#!/usr/bin/env python
# compare_models_barplots.py
# --------------------------------------------------------------------------

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# CONFIG --------------------------------------------------------------------
BASE_FILE      = Path("./7b/baseline_results.json")
FINETUNE_FILE  = Path("./finetuned/finetuned_results.json")

OUT_CONTENT_FIG = Path("compare_content_type_bar.png")
OUT_QTYPE_FIG   = Path("compare_question_type_bar.png")
# --------------------------------------------------------------------------


def load_and_prepare(fp):
    with open(fp) as f:
        recs = json.load(f)
    df = pd.DataFrame(recs)
    df.columns = [c.lower().strip() for c in df.columns]

    df["acc_part1"] = df["points_part1"] / df["max_points"]
    df["acc_part2"] = df["points_part2"] / df["max_points"]
    df["kgain"]     = df["acc_part2"]   - df["acc_part1"]

    df["content_type"] = df["content_type"].str.lower()
    df["qa_type"]      = (df["qa_type"].str.lower()
                                         .str.replace(r"[^\w]+", "_", regex=True))
    return df


def agg_micro(df, by):
    g = (df.groupby(by)
           [["points_part1", "points_part2", "max_points"]]
           .sum())
    acc1 = g["points_part1"] / g["max_points"]
    acc2 = g["points_part2"] / g["max_points"]
    gain = acc2 - acc1
    return acc1, acc2, gain


def plot_compare(acc1_b, acc2_b, gain_b,
                 acc1_f, acc2_f, gain_f,
                 groups, label_map, title, outfile):

    x = np.arange(len(groups))
    width = 0.12

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = []
    # colours: base = light tint, fine-tuned = default hue
    palette = [("#7fa9ff", "C0"), ("#ffb57f", "C1"), ("#a4d9a4", "C2")]

    # metric order: Part1 | Part2 | Gain
    series_base = [acc1_b, acc2_b, gain_b]
    series_ft   = [acc1_f, acc2_f, gain_f]

    offsets = [-2.5*width, -1.5*width, -0.5*width,
                0.5*width,  1.5*width,  2.5*width]

    metric_order = [0, 1, 2, 0, 1, 2]
    for idx, (metric_idx, off) in enumerate(zip(metric_order, offsets)):
        data = series_base[metric_idx] if idx < 3 else series_ft[metric_idx]
        color = palette[metric_idx][0] if idx < 3 else palette[metric_idx][1]
        bars.append(
            ax.bar(x + off, data.values, width=width, color=color,
                   label=None)  # legend proxies later
        )

    # numeric labels
    for bar_grp in bars:
        ax.bar_label(bar_grp, fmt="%.2f", padding=2, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([label_map[g] for g in groups], fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_ylim(0, max(series_ft[1].max(), series_base[1].max())*1.25)
    ax.set_title(title, fontsize=15, pad=12)

    # legend proxies
    legend_handles = [
        plt.Line2D([0], [0], marker="s", markersize=12, linestyle="",
                   markerfacecolor="#7fa9ff", markeredgecolor="none",
                   label="Part 1 – base"),
        plt.Line2D([0], [0], marker="s", markersize=12, linestyle="",
                   markerfacecolor="C0", markeredgecolor="none",
                   label="Part 1 – fine-tuned"),
        plt.Line2D([0], [0], marker="s", markersize=12, linestyle="",
                   markerfacecolor="#ffb57f", markeredgecolor="none",
                   label="Part 2 – base"),
        plt.Line2D([0], [0], marker="s", markersize=12, linestyle="",
                   markerfacecolor="C1", markeredgecolor="none",
                   label="Part 2 – fine-tuned"),
        plt.Line2D([0], [0], marker="s", markersize=12, linestyle="",
                   markerfacecolor="#a4d9a4", markeredgecolor="none",
                   label="KGain – base"),
        plt.Line2D([0], [0], marker="s", markersize=12, linestyle="",
                   markerfacecolor="C2", markeredgecolor="none",
                   label="KGain – fine-tuned"),
    ]
    ax.legend(handles=legend_handles, ncol=2, frameon=False, fontsize=11)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Figure saved → {outfile.resolve()}")
    plt.show()


# ---------- MAIN -----------------------------------------------------------
df_base = load_and_prepare(BASE_FILE)
df_ft   = load_and_prepare(FINETUNE_FILE)

# ——— CONTENT TYPE ----------------------------------------------------------
order_ct   = ["news", "abstract", "tweet"]
label_ct   = {"news":"News", "abstract":"Abstract", "tweet":"Tweet"}

acc1_b, acc2_b, gain_b = agg_micro(df_base, "content_type")
acc1_f, acc2_f, gain_f = agg_micro(df_ft,   "content_type")

plot_compare(acc1_b.reindex(order_ct), acc2_b.reindex(order_ct), gain_b.reindex(order_ct),
             acc1_f.reindex(order_ct), acc2_f.reindex(order_ct), gain_f.reindex(order_ct),
             groups=order_ct, label_map=label_ct,
             title="",
             outfile=OUT_CONTENT_FIG)

# ——— QUESTION TYPE ---------------------------------------------------------
order_qt = ["tf", "mc_easy", "mc_hard"]
label_qt = {"tf":"True / False", "mc_easy":"Easy MC", "mc_hard":"Hard MC"}

acc1_b, acc2_b, gain_b = agg_micro(df_base, "qa_type")
acc1_f, acc2_f, gain_f = agg_micro(df_ft,   "qa_type")

present_qt = [q for q in order_qt if q in acc1_b.index.union(acc1_f.index)]

plot_compare(acc1_b.reindex(present_qt), acc2_b.reindex(present_qt), gain_b.reindex(present_qt),
             acc1_f.reindex(present_qt), acc2_f.reindex(present_qt), gain_f.reindex(present_qt),
             groups=present_qt, label_map=label_qt,
             title="",
             outfile=OUT_QTYPE_FIG)

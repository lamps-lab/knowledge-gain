import json

with open(out_dir / "test_predictions_combined.json", "w") as f:
    json.dump(combined, f, indent=2)
print(f"Combined predictions saved → {out_dir/'test_predictions_combined.json'}")

# ----------------------------------------------------------------------
# 7. Part-level P/R/F1
# ----------------------------------------------------------------------
def prf(y_true, y_pred):
    p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r = recall_score(y_true,  y_pred, average="macro", zero_division=0)
    f = f1_score(y_true,     y_pred, average="macro", zero_division=0)
    return p, r, f

a1 = [rec["answer_part1"] for rec in combined]
a2 = [rec["answer_part2"] for rec in combined]

for part, pred in ((1, a1), (2, a2)):
    p, r, f = prf(truths, pred)
    print(f"\nPart {part} – Precision: {p:.4f}  Recall: {r:.4f}  F1: {f:.4f}")

# ----------------------------------------------------------------------
# 8. Knowledge-gain aggregation (weighted micro accuracy)
# ----------------------------------------------------------------------
df = pd.DataFrame(combined)

# content-type groups
agg_ct = (df.groupby("content_type")
            [["points_part1", "points_part2", "max_points"]].sum())
kg_ct = ((agg_ct["points_part2"] / agg_ct["max_points"])
         - (agg_ct["points_part1"] / agg_ct["max_points"])).sort_index()

# question-types
agg_qt = (df.groupby("qa_type")
            [["points_part1", "points_part2", "max_points"]].sum())
kg_qt = ((agg_qt["points_part2"] / agg_qt["max_points"])
         - (agg_qt["points_part1"] / agg_qt["max_points"])).sort_values(ascending=False)

print("\nKnowledge gain (Δ accuracy) by content type")
for k, v in kg_ct.items():
    print(f"  {k:<9}: {v:+.3f}")

print("\nKnowledge gain (Δ accuracy) by question type")
for k, v in kg_qt.items():
    print(f"  {k:<9}: {v:+.3f}")

# ---- content-type KG plot ----
ax  = kg_ct.plot.bar(
        title="Knowledge gain by content type",
        ylabel="Δ accuracy (Part 2 – Part 1)",
        ylim=(-1, 1)
      )
fig = ax.get_figure()          # grab the Figure object
fig.tight_layout()             # adjust margins
fig.savefig(out_dir / "kgain_content_type.png")
plt.close(fig)

# ---- question-type KG plot ----
ax  = kg_qt.plot.bar(
        title="Knowledge gain by question type",
        ylabel="Δ accuracy (Part 2 – Part 1)",
        ylim=(-1, 1)
      )
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(out_dir / "kgain_question_type.png")
plt.close(fig)

"""
plot_conditional_eval.py
绘制 L7 条件级评估对比图
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "conditional_eval_results.json"), "r") as f:
    report = json.load(f)

per_split = report["per_split_eval"]
conditional = report["conditional_analysis"]
protein = report["protein_level"]
module_contrib = report["module_contribution_val_both"]

# ============ 1. Per-split × model comparison (RMSE) ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("L7: Per-Split Evaluation Across Model Variants", fontsize=14, fontweight="bold")

models = ["R", "L+R", "C+L+R", "C+L+R+P"]
split_order = ["val_seen", "val_strain_only", "val_chem_only", "val_both", "val_time"]
splits = [s for s in split_order if s in per_split["R"]]

x = np.arange(len(splits))
width = 0.2
colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

for ax, metric, label in zip(axes, ["rmse", "r2"], ["RMSE (↓)", "R² (↑)"]):
    for i, mname in enumerate(models):
        vals = [per_split[mname][s][metric] for s in splits]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=mname, color=colors[i],
                      edgecolor="black", linewidth=0.3)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(label, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=20, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_L7_per_split_models.png"), dpi=150, bbox_inches="tight")
print("Saved plot_L7_per_split_models.png")
plt.close()

# ============ 2. Conditional analysis (by strain / compound / time / temp) ============
fields_to_plot = ["Strains", "pert_time", "Temperature", "Medium"]
field_titles = ["Strain", "Perturbation Time (min)", "Temperature (°C)", "Medium"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("L7: Conditional RMSE Analysis (C+L+R+P model)", fontsize=14, fontweight="bold")

for ax, field, title in zip(axes.flat, fields_to_plot, field_titles):
    if field not in conditional:
        ax.set_visible(False)
        continue
    data = conditional[field]
    items = sorted(data.items(), key=lambda x: x[1]["rmse"])
    labels = [str(k)[:20] for k, _ in items]
    rmses = [v["rmse"] for _, v in items]
    r2s = [v["r2"] for _, v in items]
    ns = [v["n_samples_in_group"] for _, v in items]

    bars = ax.bar(range(len(labels)), rmses, color="#4C72B0", edgecolor="black", linewidth=0.5)
    # Highlight best and worst
    if len(bars) > 1:
        bars[0].set_color("#55A868")  # best (lowest RMSE)
        bars[-1].set_color("#C44E52")  # worst (highest RMSE)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(f"By {title}", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    for bar, v, n in zip(bars, rmses, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.95,
                f"{v:.3f}", ha="center", va="top", fontsize=7, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_L7_conditional_rmse.png"), dpi=150, bbox_inches="tight")
print("Saved plot_L7_conditional_rmse.png")
plt.close()

# ============ 3. Per-compound RMSE (top/bottom) ============
if "perturbation_no_concentration" in conditional:
    compound_data = conditional["perturbation_no_concentration"]
    items = sorted(compound_data.items(), key=lambda x: x[1]["rmse"])

    # Show top 10 best + top 10 worst
    top_best = items[:10]
    top_worst = items[-10:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("L7: Per-Compound RMSE (Best vs Worst, C+L+R+P model)", fontsize=14, fontweight="bold")

    for ax, data_items, title, color in zip(
        axes, [top_best, top_worst],
        ["Top-10 Best (Lowest RMSE)", "Top-10 Worst (Highest RMSE)"],
        ["#55A868", "#C44E52"]
    ):
        labels = [str(k)[:18] for k, _ in data_items]
        vals = [v["rmse"] for _, v in data_items]
        bars = ax.barh(range(len(labels)), vals, color=color, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("RMSE", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plot_L7_compound_best_worst.png"), dpi=150, bbox_inches="tight")
    print("Saved plot_L7_compound_best_worst.png")
    plt.close()

# ============ 4. Module contribution on val_both (OOD) ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("L7: Module Contribution on val_both (OOD: unseen strain + chemical)",
             fontsize=14, fontweight="bold")

x = np.arange(len(models))
width = 0.35

for ax, metric, label in zip(axes, ["rmse", "r2"], ["RMSE (↓)", "R² (↑)"]):
    vals = [module_contrib[m][metric] for m in models]
    bars = ax.bar(x, vals, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(label, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.95,
                f"{v:.4f}", ha="center", va="top", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_L7_module_contribution.png"), dpi=150, bbox_inches="tight")
print("Saved plot_L7_module_contribution.png")
plt.close()

# ============ 5. Protein-level distribution ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("L7: Per-Protein RMSE Distribution (C+L+R+P model)", fontsize=14, fontweight="bold")

# Histogram of per-protein RMSE
all_protein_data = report.get("protein_level", {})
# Use stats if available
stats = all_protein_data.get("stats", {})
ax = axes[0]
# Reconstruct from best/worst is limited, so show stats
ax.barh(["Mean", "Median", "Std"], [stats.get("mean_rmse", 0), stats.get("median_rmse", 0), stats.get("std_rmse", 0)],
        color=["#4C72B0", "#55A868", "#DD8452"], edgecolor="black", linewidth=0.5)
ax.set_xlabel("RMSE", fontsize=11)
ax.set_title("Per-Protein RMSE Statistics", fontsize=12)
for i, (k, v) in enumerate(zip(["Mean", "Median", "Std"], [stats.get("mean_rmse", 0), stats.get("median_rmse", 0), stats.get("std_rmse", 0)])):
    ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

# Best vs Worst proteins
ax = axes[1]
best = all_protein_data.get("best_top15", [])[:10]
worst = all_protein_data.get("worst_top15", [])[:10]
protein_names = [r["protein"] for r in best] + [r["protein"] for r in worst]
protein_rmses = [r["rmse"] for r in best] + [r["rmse"] for r in worst]
protein_colors = ["#55A868"] * len(best) + ["#C44E52"] * len(worst)

bars = ax.barh(range(len(protein_names)), protein_rmses, color=protein_colors,
               edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(protein_names)))
ax.set_yticklabels(protein_names, fontsize=7)
ax.set_xlabel("RMSE", fontsize=11)
ax.set_title("Top-10 Best (green) vs Top-10 Worst (red) Proteins", fontsize=12)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_L7_protein_level.png"), dpi=150, bbox_inches="tight")
print("Saved plot_L7_protein_level.png")
plt.close()

print("\nAll L7 plots saved!")

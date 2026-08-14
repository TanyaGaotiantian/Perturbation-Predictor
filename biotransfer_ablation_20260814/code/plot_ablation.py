"""
plot_ablation.py
绘制消融实验对比图
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "ablation_results.json"), "r") as f:
    report = json.load(f)

results = report["results"]

# ============ 1. L4: MLP vs MLP+R (4 metrics) ============
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("L4: MLP vs MLP+Residual (4 Core Metrics)", fontsize=14, fontweight="bold")

l4_names = ["MLP", "MLP+R"]
l4_colors = ["#4C72B0", "#DD8452"]
metrics = ["rmse", "r2", "mae", "pearson_r"]
metric_labels = ["RMSE (↓)", "R² (↑)", "MAE (↓)", "Pearson r (↑)"]

for ax, metric, label in zip(axes, metrics, metric_labels):
    vals = [results[n][metric] for n in l4_names]
    bars = ax.bar(l4_names, vals, color=l4_colors, width=0.5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=12)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.95,
                f"{v:.4f}", ha="center", va="top", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_L4_residual_comparison.png"), dpi=150, bbox_inches="tight")
print("Saved plot_L4_residual_comparison.png")
plt.close()

# ============ 2. L5: Interaction comparison ============
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("L5: Interaction Module Comparison (4 Core Metrics)", fontsize=14, fontweight="bold")

l5_names = ["Baseline_MLP", "Hadamard", "InteractMLP"]
l5_labels = ["Baseline\n(strain+drug)", "Hadamard\n(strain⊙drug)", "Interaction\nMLP"]
l5_colors = ["#4C72B0", "#55A868", "#C44E52"]

for ax, metric, label in zip(axes, metrics, metric_labels):
    vals = [results[n][metric] for n in l5_names]
    bars = ax.bar(l5_labels, vals, color=l5_colors, width=0.5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=12)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.95,
                f"{v:.4f}", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_L5_interaction_comparison.png"), dpi=150, bbox_inches="tight")
print("Saved plot_L5_interaction_comparison.png")
plt.close()

# ============ 3. L5: Ablation progression ============
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("L5: Progressive Ablation (MLP → R → L+R → C+L+R → C+L+R+P)", fontsize=14, fontweight="bold")

abl_names = ["ablation_MLP", "ablation_R", "ablation_L+R", "ablation_C+L+R", "ablation_C+L+R+P"]
abl_labels = ["MLP", "+R", "L+R", "C+L+R", "C+L+R+P"]
abl_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

for ax, metric, label in zip(axes, metrics, metric_labels):
    vals = [results[n][metric] for n in abl_names]
    bars = ax.bar(abl_labels, vals, color=abl_colors, width=0.6, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=12)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.95,
                f"{v:.4f}", ha="center", va="top", fontsize=7, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_L5_ablation_progressive.png"), dpi=150, bbox_inches="tight")
print("Saved plot_L5_ablation_progressive.png")
plt.close()

# ============ 4. All experiments overview ============
fig, ax = plt.subplots(figsize=(14, 6))
all_names = ["MLP", "MLP+R", "Hadamard", "InteractMLP",
             "ablation_L+R", "ablation_C+L+R", "ablation_C+L+R+P"]
all_labels = ["MLP", "MLP+R", "Hadamard", "InteractMLP",
              "L+R", "C+L+R", "C+L+R+P"]

x = np.arange(len(all_names))
width = 0.2

rmse_vals = [results[n]["rmse"] for n in all_names]
r2_vals = [results[n]["r2"] for n in all_names]
mae_vals = [results[n]["mae"] for n in all_names]
pr_vals = [results[n]["pearson_r"] for n in all_names]

# Normalize to 0-1 for comparison
rmse_norm = 1 - (np.array(rmse_vals) - min(rmse_vals)) / (max(rmse_vals) - min(rmse_vals) + 1e-8)
r2_norm = (np.array(r2_vals) - min(r2_vals)) / (max(r2_vals) - min(r2_vals) + 1e-8)
mae_norm = 1 - (np.array(mae_vals) - min(mae_vals)) / (max(mae_vals) - min(mae_vals) + 1e-8)
pr_norm = (np.array(pr_vals) - min(pr_vals)) / (max(pr_vals) - min(pr_vals) + 1e-8)

bars1 = ax.bar(x - 1.5*width, rmse_norm, width, label="RMSE (normalized, ↑better)", color="#4C72B0")
bars2 = ax.bar(x - 0.5*width, r2_norm, width, label="R² (normalized)", color="#55A868")
bars3 = ax.bar(x + 0.5*width, mae_norm, width, label="MAE (normalized, ↑better)", color="#DD8452")
bars4 = ax.bar(x + 1.5*width, pr_norm, width, label="Pearson r (normalized)", color="#C44E52")

ax.set_xlabel("Model Variant", fontsize=12)
ax.set_ylabel("Normalized Score (1=best, 0=worst)", fontsize=12)
ax.set_title("All Experiments: Normalized Metric Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(all_labels, rotation=30, ha="right")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(-0.05, 1.15)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_all_experiments_overview.png"), dpi=150, bbox_inches="tight")
print("Saved plot_all_experiments_overview.png")
plt.close()

# ============ 5. L7: Per-split evaluation ============
if "per_split_eval" in report and report["per_split_eval"]:
    split_data = report["per_split_eval"]
    split_names = list(split_data.keys())
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("L7: Per-Split Evaluation (C+L+R+P Model)", fontsize=14, fontweight="bold")
    
    split_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        vals = [split_data[sn][metric] for sn in split_names]
        bars = ax.bar(split_names, vals, color=split_colors[:len(split_names)],
                      width=0.5, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.95,
                    f"{v:.4f}", ha="center", va="top", fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plot_L7_per_split_eval.png"), dpi=150, bbox_inches="tight")
    print("Saved plot_L7_per_split_eval.png")
    plt.close()

print("\nAll plots saved!")

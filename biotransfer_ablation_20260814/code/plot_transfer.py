"""
plot_transfer.py
绘制 L6 跨数据迁移实验对比图
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "transfer_results.json"), "r") as f:
    report = json.load(f)

results = report["results"]
split_eval = report["per_split_eval"]

metrics = ["rmse", "r2", "mae", "pearson_r"]
metric_labels = ["RMSE (↓)", "R² (↑)", "MAE (↓)", "Pearson r (↑)"]

# ============ 1. Overall: scratch vs pretrained (4 metrics) ============
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("L6: Cross-dataset Transfer — Scratch vs Pretrained (Overall)", fontsize=14, fontweight="bold")

names = ["scratch", "pretrained"]
labels = ["Scratch\n(from zero)", "Pretrained\n(WAYB→WAYC)"]
colors = ["#4C72B0", "#DD8452"]

for ax, metric, label in zip(axes, metrics, metric_labels):
    vals = [results[n][metric] for n in names]
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=12)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.95,
                f"{v:.4f}", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_L6_overall_comparison.png"), dpi=150, bbox_inches="tight")
print("Saved plot_L6_overall_comparison.png")
plt.close()

# ============ 2. Per-split: scratch vs pretrained ============
split_names = list(split_eval["scratch"].keys())
# Order: val_seen-like first, then OOD
preferred_order = ["val_seen", "val_strain_only", "val_chem_only", "val_both", "val_time"]
split_names = [s for s in preferred_order if s in split_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("L6: Per-Split Transfer Comparison (Scratch vs Pretrained)", fontsize=14, fontweight="bold")

x = np.arange(len(split_names))
width = 0.35

for ax, metric, label in zip(axes, ["rmse", "r2"], ["RMSE (↓)", "R² (↑)"]):
    scratch_vals = [split_eval["scratch"][s][metric] for s in split_names]
    pre_vals = [split_eval["pretrained"][s][metric] for s in split_names]

    bars1 = ax.bar(x - width/2, scratch_vals, width, label="Scratch", color="#4C72B0",
                   edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, pre_vals, width, label="Pretrained", color="#DD8452",
                   edgecolor="black", linewidth=0.5)

    for bar, v in zip(bars1, scratch_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.95,
                f"{v:.3f}", ha="center", va="top", fontsize=8, fontweight="bold")
    for bar, v in zip(bars2, pre_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.95,
                f"{v:.3f}", ha="center", va="top", fontsize=8, fontweight="bold")

    ax.set_ylabel(label, fontsize=12)
    ax.set_title(label, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(split_names, rotation=20, ha="right")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_L6_per_split_comparison.png"), dpi=150, bbox_inches="tight")
print("Saved plot_L6_per_split_comparison.png")
plt.close()

# ============ 3. Per-split improvement (ΔRMSE) ============
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("L6: Pretrained Improvement over Scratch (ΔRMSE, positive = pretrained better)",
             fontsize=13, fontweight="bold")

deltas = [split_eval["scratch"][s]["rmse"] - split_eval["pretrained"][s]["rmse"]
           for s in split_names]
colors_bar = ["#55A868" if d > 0 else "#C44E52" for d in deltas]

bars = ax.bar(split_names, deltas, color=colors_bar, width=0.5, edgecolor="black", linewidth=0.5)
ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
ax.set_ylabel("ΔRMSE (scratch − pretrained)", fontsize=12)
ax.set_xlabel("Validation Split", fontsize=12)
for bar, v in zip(bars, deltas):
    y_pos = bar.get_height() + 0.0005 if v >= 0 else bar.get_height() - 0.001
    va = "bottom" if v >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{v:+.4f}",
            ha="center", va=va, fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_L6_improvement_delta.png"), dpi=150, bbox_inches="tight")
print("Saved plot_L6_improvement_delta.png")
plt.close()

# ============ 4. Training curves ============
hist_path = os.path.join(BASE_DIR, "transfer_history.json")
if os.path.exists(hist_path):
    with open(hist_path, "r") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("L6: Training Curves", fontsize=14, fontweight="bold")

    # Scratch train loss + val RMSE
    if "scratch" in history:
        h = history["scratch"]
        eps = [x["epoch"] for x in h]
        ax = axes[0]
        ax.plot(eps, [x["train_loss"] for x in h], "o-", color="#4C72B0", label="Train Loss")
        ax2 = ax.twinx()
        ax2.plot(eps, [x["rmse"] for x in h], "s--", color="#C44E52", label="Val RMSE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss", color="#4C72B0")
        ax2.set_ylabel("Val RMSE", color="#C44E52")
        ax.set_title("Scratch (WAYC train only)", fontsize=12)

    # Pretrained: pretrain + finetune
    ax = axes[1]
    pre_h = history.get("pretrained_pretrain", [])
    fine_h = history.get("pretrained_finetune", [])
    if pre_h:
        eps_pre = [x["epoch"] for x in pre_h]
        ax.plot(eps_pre, [x["train_loss"] for x in pre_h], "o-", color="#55A868",
                label="Pretrain Loss (WAYB)")
    if fine_h:
        eps_fine = [x["epoch"] for x in fine_h]
        offset = len(pre_h)
        ax.plot([e + offset for e in eps_fine], [x["train_loss"] for x in fine_h], "s-",
                color="#DD8452", label="Finetune Loss (WAYC)")
    ax.set_xlabel("Epoch (pretrain | finetune)")
    ax.set_ylabel("Train Loss")
    ax.set_title("Pretrained (WAYB → WAYC)", fontsize=12)
    ax.legend(fontsize=9)
    ax.axvline(x=len(pre_h) + 0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plot_L6_training_curves.png"), dpi=150, bbox_inches="tight")
    print("Saved plot_L6_training_curves.png")
    plt.close()

print("\nAll L6 plots saved!")

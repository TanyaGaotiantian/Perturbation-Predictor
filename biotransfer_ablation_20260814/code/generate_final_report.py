"""
generate_final_report.py
L8: 最终收敛 — 生成主结果表/消融表/OOD表/迁移表/架构图

聚合所有实验结果:
  - ablation_results.json      (L4/L5 消融 + per-split)
  - transfer_results.json      (L6 跨数据迁移)
  - conditional_eval_results.json (L7 条件级评估)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============ Load all results ============
with open(os.path.join(BASE_DIR, "ablation_results.json"), "r") as f:
    abl = json.load(f)
with open(os.path.join(BASE_DIR, "transfer_results.json"), "r") as f:
    transfer = json.load(f)
with open(os.path.join(BASE_DIR, "conditional_eval_results.json"), "r") as f:
    cond = json.load(f)

report_lines = []

def L(s=""):
    report_lines.append(s)
    print(s)


# ============ Table 1: Main Results (all methods) ============
L("=" * 90)
L("TABLE 1: Main Results — All Methods (val set, 4 core metrics)")
L("=" * 90)
L(f"{'Method':<22} {'RMSE(↓)':<10} {'R²(↑)':<10} {'MAE(↓)':<10} {'Pearson(↑)':<10} {'Params':<12}")
L("-" * 74)

main_methods = [
    ("MLP (baseline)",        abl["results"]["MLP"]),
    ("MLP+R",                 abl["results"]["MLP+R"]),
    ("Hadamard (strain⊙drug)", abl["results"]["Hadamard"]),
    ("Interaction MLP",       abl["results"]["InteractMLP"]),
]
for name, r in main_methods:
    L(f"{name:<22} {r['rmse']:<10.4f} {r['r2']:<10.4f} {r['mae']:<10.4f} "
      f"{r['pearson_r']:<10.4f} {r['num_params']:<12,}")

L("")

# ============ Table 2: Progressive Ablation ============
L("=" * 90)
L("TABLE 2: Progressive Ablation (MLP → R → L+R → C+L+R → C+L+R+P)")
L("=" * 90)
L(f"{'Variant':<18} {'Components':<28} {'RMSE(↓)':<10} {'R²(↑)':<10} {'MAE(↓)':<10} {'Pearson(↑)':<10}")
L("-" * 86)

abl_methods = [
    ("MLP",     "base",                 "ablation_MLP"),
    ("+R",      "Residual",             "ablation_R"),
    ("L+R",     "Latent + Residual",    "ablation_L+R"),
    ("C+L+R",   "CrossAttn + L + R",   "ablation_C+L+R"),
    ("C+L+R+P", "CrossAttn+L+R+Prior", "ablation_C+L+R+P"),
]
for label, comp, key in abl_methods:
    r = abl["results"][key]
    L(f"{label:<18} {comp:<28} {r['rmse']:<10.4f} {r['r2']:<10.4f} {r['mae']:<10.4f} {r['pearson_r']:<10.4f}")

# Find best
best_abl = min(abl_methods, key=lambda x: abl["results"][x[2]]["rmse"])
L(f"\nBest ablation variant: {best_abl[0]} ({best_abl[1]}) — RMSE={abl['results'][best_abl[2]]['rmse']:.4f}")

L("")

# ============ Table 3: OOD (per-split) Evaluation ============
L("=" * 90)
L("TABLE 3: OOD Evaluation — Per-Split RMSE (from conditional_eval, 4 model variants)")
L("=" * 90)

split_order = ["val_seen", "val_strain_only", "val_chem_only", "val_both", "val_time"]
models_cond = ["R", "L+R", "C+L+R", "C+L+R+P"]

# RMSE table
header = f"{'Split':<20}"
for m in models_cond:
    header += f" {m:<14}"
L(header)

available_splits = [s for s in split_order if s in cond["per_split_eval"]["R"]]

for sn in available_splits:
    line = f"{sn:<20}"
    for m in models_cond:
        if sn in cond["per_split_eval"][m]:
            v = cond["per_split_eval"][m][sn]["rmse"]
            line += f" {v:<14.4f}"
        else:
            line += f" {'N/A':<14}"
    L(line)

# R² table
header2 = f"\n{'Split':<20}"
for m in models_cond:
    header2 += f" {'R²('+m+')':<14}"
L(header2)
for sn in available_splits:
    line = f"{sn:<20}"
    for m in models_cond:
        if sn in cond["per_split_eval"][m]:
            v = cond["per_split_eval"][m][sn]["r2"]
            line += f" {v:<14.4f}"
        else:
            line += f" {'N/A':<14}"
    L(line)

L("")

# ============ Table 4: Cross-dataset Transfer ============
L("=" * 90)
L("TABLE 4: Cross-dataset Transfer Learning (L6: WAYB pretrain → WAYC finetune)")
L("=" * 90)
L(f"{'Method':<15} {'RMSE(↓)':<10} {'R²(↑)':<10} {'MAE(↓)':<10} {'Pearson(↑)':<10}")
L("-" * 55)
for name in ["scratch", "pretrained"]:
    r = transfer["results"][name]
    L(f"{name:<15} {r['rmse']:<10.4f} {r['r2']:<10.4f} {r['mae']:<10.4f} {r['pearson_r']:<10.4f}")

s_rmse = transfer["results"]["scratch"]["rmse"]
p_rmse = transfer["results"]["pretrained"]["rmse"]
delta = s_rmse - p_rmse
pct = (delta / s_rmse) * 100
L(f"\nΔRMSE = {delta:+.4f} ({pct:+.2f}%) — pretrained {'BETTER' if delta > 0 else 'WORSE'}")

L(f"\nPer-Split Transfer Comparison (RMSE):")
L(f"{'Split':<20} {'scratch':<12} {'pretrained':<12} {'ΔRMSE':<10} {'Winner':<10}")
L("-" * 64)
for sn in ["val_both", "val_strain_only", "val_chem_only", "val_time"]:
    if sn in transfer["per_split_eval"]["scratch"]:
        s = transfer["per_split_eval"]["scratch"][sn]["rmse"]
        p = transfer["per_split_eval"]["pretrained"][sn]["rmse"]
        d = s - p
        winner = "pretrained" if d > 0 else "scratch"
        L(f"{sn:<20} {s:<12.4f} {p:<12.4f} {d:<+10.4f} {winner:<10}")

L("")

# ============ Table 5: Conditional Analysis Summary ============
L("=" * 90)
L("TABLE 5: Conditional Analysis Summary (C+L+R+P model, val set)")
L("=" * 90)

for field, title in [("Strains", "Strain"), ("pert_time", "Perturbation Time"),
                      ("Temperature", "Temperature"), ("Medium", "Medium")]:
    if field in cond["conditional_analysis"]:
        data = cond["conditional_analysis"][field]
        items = sorted(data.items(), key=lambda x: x[1]["rmse"])
        L(f"\n  By {title}:")
        L(f"    {'Value':<30} {'n':<6} {'RMSE':<10} {'R²':<10}")
        for gval, m in items:
            L(f"    {str(gval)[:28]:<30} {m['n_samples_in_group']:<6} {m['rmse']:<10.4f} {m['r2']:<10.4f}")
        best = items[0]
        worst = items[-1]
        L(f"    → Best: {best[0]} (RMSE={best[1]['rmse']:.4f})")
        L(f"    → Worst: {worst[0]} (RMSE={worst[1]['rmse']:.4f})")

# ============ Table 6: Module Contribution on OOD ============
L("")
L("=" * 90)
L("TABLE 6: Module Contribution on val_both (OOD: unseen strain + chemical)")
L("=" * 90)
L(f"{'Model':<12} {'RMSE(↓)':<10} {'R²(↑)':<10} {'MAE(↓)':<10} {'Pearson(↑)':<10}")
L("-" * 52)
for m in models_cond:
    r = cond["module_contribution_val_both"][m]
    L(f"{m:<12} {r['rmse']:<10.4f} {r['r2']:<10.4f} {r['mae']:<10.4f} {r['pearson_r']:<10.4f}")

base = cond["module_contribution_val_both"]["C+L+R+P"]["rmse"]
L(f"\nDeltas vs C+L+R+P (positive ΔRMSE = removing module HURTS, i.e. module helps):")
for m in ["R", "L+R", "C+L+R"]:
    v = cond["module_contribution_val_both"][m]["rmse"]
    d = v - base
    helps = "module HELPS" if d > 0 else "module HURTS"
    L(f"  Remove → {m:<10}: ΔRMSE = {d:+.4f}  ({helps})")

# ============ Table 7: Protein-level Summary ============
L("")
L("=" * 90)
L("TABLE 7: Protein-Level Analysis Summary (C+L+R+P model, val set)")
L("=" * 90)
stats = cond["protein_level"]["stats"]
L(f"  Proteins evaluated:  {stats['n_proteins_evaluated']}")
L(f"  Mean per-protein RMSE:    {stats['mean_rmse']:.4f}")
L(f"  Median per-protein RMSE:  {stats['median_rmse']:.4f}")
L(f"  Std per-protein RMSE:     {stats['std_rmse']:.4f}")
L(f"  Mean per-protein Pearson: {stats['mean_pearson']:.4f}")

L(f"\n  Top-5 BEST predicted proteins:")
L(f"    {'Protein':<15} {'RMSE':<10} {'Pearson':<10}")
for r in cond["protein_level"]["best_top15"][:5]:
    L(f"    {r['protein'][:13]:<15} {r['rmse']:<10.4f} {r['pearson_r']:<10.4f}")

L(f"\n  Top-5 WORST predicted proteins:")
L(f"    {'Protein':<15} {'RMSE':<10} {'Pearson':<10}")
for r in cond["protein_level"]["worst_top15"][:5]:
    L(f"    {r['protein'][:13]:<15} {r['rmse']:<10.4f} {r['pearson_r']:<10.4f}")

# ============ Save text report ============
report_path = os.path.join(BASE_DIR, "final_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\nReport saved to {report_path}")


# ============ Architecture Diagram ============
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("BioTransfer-VCell: Architecture Diagram (C+L+R+P + Transfer)",
             fontsize=16, fontweight="bold", pad=20)

def box(x, y, w, h, text, color="#4C72B0", fontsize=9, alpha=0.85):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor="black", linewidth=1.5, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", wrap=True)

def arrow(x1, y1, x2, y2, label="", color="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.2, label, ha="center", fontsize=7, color=color, fontstyle="italic")

# Input layer
box(0.5, 8.0, 2.5, 1.2, "Protein Input\n(x, mask)\n4422 proteins", "#4C72B0")
box(0.5, 6.0, 2.5, 1.2, "Strain Embedding\n(learned, 32d)", "#55A868")
box(0.5, 4.5, 2.5, 1.2, "Medium/Temp\n(learned emb)", "#55A868")
box(0.5, 3.0, 2.5, 1.2, "Chemical (Morgan FP\n1024-bit, r=2)", "#55A868")

# Encoder layer
box(4.0, 8.0, 2.5, 1.2, "Protein Encoder\n(MLP→latent 512d)", "#DD8452")
box(4.0, 5.0, 2.5, 1.2, "Context Encoder\n(MLP→latent 512d)", "#DD8452")

# Cross-attention
box(8.0, 7.5, 2.5, 1.5, "Cross-Attention\n(C)\nprotein Q × context KV", "#C44E52")

# Latent fusion
box(8.0, 5.0, 2.5, 1.2, "Latent Fusion (L)\nconcat→MLP→512d", "#8172B3")

# Decoder
box(12.0, 6.0, 2.5, 1.5, "Decoder MLP\n(1024→1024→4422)", "#4C72B0")

# Residual
box(12.0, 8.2, 2.5, 1.0, "Residual (R)\ngate·x (per-protein)", "#55A868")

# Protein Prior
box(12.0, 3.8, 2.5, 1.2, "Protein Prior (P)\nbias + low-rank", "#C44E52")

# Output
box(15.5, 5.5, 2.0, 1.5, "Output\nΔprotein\n(4422d)", "#333333")

# Transfer learning
box(0.5, 0.8, 5.0, 1.5, "Transfer Learning (L6)\nWAYB pretrain → WAYC finetune\n(dataset-specific z-score)", "#9467BD", fontsize=8)

# Arrows
arrow(3.0, 8.6, 4.0, 8.6)
arrow(3.0, 6.6, 4.0, 5.6, "strain")
arrow(3.0, 5.1, 4.0, 5.4, "med/temp")
arrow(3.0, 3.6, 4.0, 5.2, "chem")
arrow(6.5, 8.6, 8.0, 8.2, "protein_z")
arrow(6.5, 5.6, 8.0, 5.6, "context_z")
arrow(8.0, 8.2, 10.5, 7.0)
arrow(10.5, 7.0, 12.0, 7.0)
arrow(10.5, 5.6, 12.0, 6.5, "fused z")
arrow(14.5, 7.0, 15.5, 6.2)
arrow(3.0, 8.2, 12.0, 8.7, "x (skip)")
arrow(14.5, 8.7, 15.5, 6.5)
arrow(14.5, 6.0, 12.0, 4.4, "preds")
arrow(14.5, 3.8, 15.5, 5.8, "prior")

# Legend
ax.text(9.0, 1.0, "Components: MLP (base) → +R (residual) → +L (latent) → +C (cross-attn) → +P (prior) → +Transfer (L6)",
        ha="center", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_L8_architecture.png"), dpi=150, bbox_inches="tight")
print("Saved plot_L8_architecture.png")
plt.close()

# ============ Save summary JSON ============
summary = {
    "table1_main_results": {name: {k: v for k, v in r.items() if k != "history"}
                            for name, r in abl["results"].items()},
    "table2_ablation": {label: {k: v for k, v in abl["results"][key].items() if k != "history"}
                         for label, _, key in abl_methods},
    "table3_ood_per_split": cond["per_split_eval"],
    "table4_transfer": {
        "overall": {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                     for k, v in transfer["results"].items()},
        "per_split": transfer["per_split_eval"],
    },
    "table5_conditional": cond["conditional_analysis"],
    "table6_module_contribution": cond["module_contribution_val_both"],
    "table7_protein_level": cond["protein_level"],
    "best_model": {
        "ablation_best": best_abl[0],
        "ablation_best_rmse": abl["results"][best_abl[2]]["rmse"],
        "transfer_pretrained_rmse": transfer["results"]["pretrained"]["rmse"],
        "transfer_improvement_pct": float(pct),
    },
}

summary_path = os.path.join(BASE_DIR, "final_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Summary saved to {summary_path}")

print("\n" + "=" * 60)
print("L8 FINAL REPORT GENERATED")
print("=" * 60)
print(f"  - final_report.txt         (all tables)")
print(f"  - final_summary.json       (structured data)")
print(f"  - plot_L8_architecture.png (architecture diagram)")

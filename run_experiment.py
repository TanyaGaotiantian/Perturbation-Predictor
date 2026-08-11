import os
import sys
import json
import yaml
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


def log(msg, log_file=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {msg}"
    print(line)
    if log_file:
        log_file.write(line + "\n")
        log_file.flush()


def compute_metrics(y_true, y_pred):
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_v = y_true[valid_mask]
    y_pred_v = y_pred[valid_mask]

    rmse = np.sqrt(np.mean((y_true_v - y_pred_v) ** 2))
    r2 = r2_score(y_true_v, y_pred_v)
    pearson_r, _ = pearsonr(y_true_v, y_pred_v)

    return {
        "rmse": float(rmse),
        "r2_score": float(r2),
        "pearson_r": float(pearson_r),
        "num_valid": int(valid_mask.sum()),
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = "baseline_comparison"
    exp_idx = 1
    while True:
        exp_dir_name = f"exp_{exp_idx:03d}_{experiment_name}_{timestamp}"
        exp_dir = os.path.join(base_dir, "experiments", exp_dir_name)
        if not os.path.exists(exp_dir):
            break
        exp_idx += 1

    os.makedirs(exp_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, "log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def L(msg):
        log(msg, log_file)

    L("=" * 60)
    L(f"Experiment Directory: {exp_dir}")
    L("=" * 60)

    config = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "data_paths": {
            "metadata_train_val": "WAYB_WAYC_metadata_train_val(1).csv",
            "metadata_test": "WAYB_WAYC_metadata_test(1).csv",
            "proteome_train_val": "WAYB_WAYC_proteome_raw_train_val.csv",
            "proteome_test": "WAYB_WAYC_proteome_raw_test.csv",
        },
        "algorithms": ["Baseline 1: Global Protein Mean", "Baseline 2: Control Baseline with 4-level Backoff"],
        "filtering": {
            "max_missing_rate": 0.80,
            "filter_on_train_only": True,
        },
        "normalization": "log2 transform",
        "control_chemicals": ["DMSO", "Water", "Quality Control"],
        "backoff_levels": {
            "level_1": "strain + medium + temperature (strict match)",
            "level_2": "strain only",
            "level_3": "global control mean",
            "level_4": "global protein mean (Baseline 1 fallback)",
        },
        "evaluation_metrics": ["RMSE", "R2 Score", "Pearson Correlation"],
    }

    with open(os.path.join(exp_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    L("Config saved to config.yaml")

    # ============================================================
    # Step 1: Data Loading and Alignment
    # ============================================================
    L("\n" + "=" * 60)
    L("Step 1: Data Loading and Alignment")
    L("=" * 60)

    meta_train_val = pd.read_csv(os.path.join(base_dir, "WAYB_WAYC_metadata_train_val(1).csv"))
    meta_test = pd.read_csv(os.path.join(base_dir, "WAYB_WAYC_metadata_test(1).csv"))
    meta = pd.concat([meta_train_val, meta_test], ignore_index=True)

    protein_train_val = pd.read_csv(os.path.join(base_dir, "WAYB_WAYC_proteome_raw_train_val.csv"))
    protein_test = pd.read_csv(os.path.join(base_dir, "WAYB_WAYC_proteome_raw_test.csv"))
    protein = pd.concat([protein_train_val, protein_test], ignore_index=True)

    L(f"meta shape: {meta.shape}")
    L(f"protein shape: {protein.shape}")

    meta = meta.set_index("sample_ID")
    protein = protein.set_index("sample_ID")

    common_ids = meta.index.intersection(protein.index)
    meta = meta.loc[common_ids]
    protein = protein.loc[common_ids]

    L(f"对齐后总样本数: {len(common_ids)}")

    # ============================================================
    # Step 2: Feature Filtering (on train set only)
    # ============================================================
    L("\n" + "=" * 60)
    L("Step 2: Feature Filtering (train set only, anti-leakage)")
    L("=" * 60)

    train_mask = meta["split_final"] == "train"
    train_protein = protein.loc[train_mask]

    missing_rate = train_protein.isna().mean()
    valid_protein_cols = missing_rate[missing_rate < 0.8].index
    L(f"保留的蛋白列数量 (缺失率 < 80%): {len(valid_protein_cols)}")

    protein_filtered = protein[valid_protein_cols]

    # ============================================================
    # Step 3: log2 Transform
    # ============================================================
    L("\n" + "=" * 60)
    L("Step 3: log2 Transform")
    L("=" * 60)

    protein_log2 = np.log2(protein_filtered)
    L(f"原始 NaN 数量: {protein_filtered.isna().sum().sum()}")
    L(f"log2 后 NaN 数量: {protein_log2.isna().sum().sum()}")
    L(f"NaN 保持一致: {protein_filtered.isna().sum().sum() == protein_log2.isna().sum().sum()}")

    # ============================================================
    # Step 4: Train / Validation Split
    # ============================================================
    L("\n" + "=" * 60)
    L("Step 4: Train / Validation Split")
    L("=" * 60)

    train_log2 = protein_log2.loc[train_mask]
    val_mask = meta["split_final"].str.startswith("val")
    val_log2 = protein_log2.loc[val_mask]
    val_meta = meta.loc[val_mask]

    L(f"训练集样本数: {train_log2.shape[0]}")
    L(f"验证集样本数: {val_log2.shape[0]}")
    L(f"蛋白数量: {val_log2.shape[1]}")

    # ============================================================
    # Step 5: Baseline 1 - Global Protein Mean
    # ============================================================
    L("\n" + "=" * 60)
    L("Step 5: Baseline 1 - Global Protein Mean Baseline")
    L("=" * 60)

    protein_mean_vector = train_log2.mean(axis=0, skipna=True)

    val_preds_b1 = pd.DataFrame(
        np.tile(protein_mean_vector.values, (val_log2.shape[0], 1)),
        index=val_log2.index,
        columns=val_log2.columns,
    )

    val_true = val_log2

    L(f"验证集预测矩阵 shape: {val_preds_b1.shape}")

    b1_metrics = compute_metrics(
        val_true.values.flatten(),
        val_preds_b1.values.flatten(),
    )

    L(f"有效评估位置数 (非缺失): {b1_metrics['num_valid']}")
    L(f"Baseline 1 RMSE: {b1_metrics['rmse']:.6f}")
    L(f"Baseline 1 R2 Score: {b1_metrics['r2_score']:.6f}")
    L(f"Baseline 1 Pearson r: {b1_metrics['pearson_r']:.6f}")

    # ============================================================
    # Step 6: Baseline 2 - Control Backoff
    # ============================================================
    L("\n" + "=" * 60)
    L("Step 6: Baseline 2 - Control Baseline with 4-level Backoff")
    L("=" * 60)

    control_chemicals = ["DMSO", "Water", "Quality Control"]
    train_meta = meta.loc[train_mask]
    is_control = train_meta["perturbation_no_concentration"].isin(control_chemicals)
    train_control_meta = train_meta.loc[is_control]
    train_control_log2 = protein_log2.loc[train_control_meta.index]

    L(f"训练集 Control 样本数: {len(train_control_meta)}")
    L(f"Control 化学物分布:\n{train_control_meta['perturbation_no_concentration'].value_counts()}")
    L(f"Control 涉及的菌株: {sorted(train_control_meta['Strains'].unique())}")

    # Precompute backoff levels
    ctrl_lvl1 = train_control_log2.groupby(
        [train_control_meta["Strains"], train_control_meta["Medium"], train_control_meta["Temperature"]]
    ).mean()
    L(f"Level 1 组合数 (strain×medium×temp): {len(ctrl_lvl1)}")

    ctrl_lvl2 = train_control_log2.groupby(train_control_meta["Strains"]).mean()
    L(f"Level 2 组合数 (strain): {len(ctrl_lvl2)}")

    ctrl_lvl3 = train_control_log2.mean(skipna=True)
    L(f"Level 3 全局 Control 向量长度: {len(ctrl_lvl3)}")

    ctrl_lvl4 = protein_mean_vector
    L(f"Level 4 = Baseline 1 全局均值向量长度: {len(ctrl_lvl4)}")

    # Build predictions with backoff
    n_val = val_log2.shape[0]
    n_proteins = val_log2.shape[1]

    val_preds_b2 = pd.DataFrame(
        np.full((n_val, n_proteins), np.nan),
        index=val_log2.index,
        columns=val_log2.columns,
    )

    level_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for i, (sid, row) in enumerate(val_meta.iterrows()):
        strain = row["Strains"]
        medium = row["Medium"]
        temp = row["Temperature"]
        pred_vec = None
        used_level = None

        if (strain, medium, temp) in ctrl_lvl1.index:
            pred_vec = ctrl_lvl1.loc[(strain, medium, temp)].values
            used_level = 1
        elif strain in ctrl_lvl2.index:
            pred_vec = ctrl_lvl2.loc[strain].values
            used_level = 2
        elif len(train_control_log2) > 0:
            pred_vec = ctrl_lvl3.values
            used_level = 3
        else:
            pred_vec = ctrl_lvl4.values
            used_level = 4

        val_preds_b2.iloc[i] = pred_vec
        level_counts[used_level] += 1

    # Fill remaining NaN in prediction matrix
    val_preds_b2 = val_preds_b2.fillna(ctrl_lvl3)
    val_preds_b2 = val_preds_b2.fillna(ctrl_lvl4)

    L("\n===== 验证集 Control 匹配统计 =====")
    for lvl in [1, 2, 3, 4]:
        cnt = level_counts[lvl]
        pct = cnt / n_val * 100
        L(f"Level {lvl}: {cnt} 样本 ({pct:.2f}%)")

    b2_metrics = compute_metrics(
        val_true.values.flatten(),
        val_preds_b2.values.flatten(),
    )

    L(f"\n有效评估位置数 (非缺失): {b2_metrics['num_valid']}")
    L(f"Baseline 2 RMSE: {b2_metrics['rmse']:.6f}")
    L(f"Baseline 2 R2 Score: {b2_metrics['r2_score']:.6f}")
    L(f"Baseline 2 Pearson r: {b2_metrics['pearson_r']:.6f}")

    # ============================================================
    # Step 7: Comparison & Summary
    # ============================================================
    L("\n" + "=" * 60)
    L("Step 7: Baseline 1 vs Baseline 2 Comparison")
    L("=" * 60)

    rmse_diff = b2_metrics["rmse"] - b1_metrics["rmse"]
    r2_diff = b2_metrics["r2_score"] - b1_metrics["r2_score"]
    pearson_diff = b2_metrics["pearson_r"] - b1_metrics["pearson_r"]

    rmse_pct = (rmse_diff / b1_metrics["rmse"]) * 100
    r2_pct = (r2_diff / abs(b1_metrics["r2_score"])) * 100 if b1_metrics["r2_score"] != 0 else float("nan")
    pearson_pct = (pearson_diff / abs(b1_metrics["pearson_r"])) * 100 if b1_metrics["pearson_r"] != 0 else float("nan")

    header = f"{'指标':<20} {'Baseline 1':<18} {'Baseline 2':<18} {'变化':<18}"
    L(header)
    L("-" * 70)
    L(f"{'RMSE':<20} {b1_metrics['rmse']:<18.6f} {b2_metrics['rmse']:<18.6f} {rmse_diff:+.6f} ({rmse_pct:+.2f}%)")
    L(f"{'R2 Score':<20} {b1_metrics['r2_score']:<18.6f} {b2_metrics['r2_score']:<18.6f} {r2_diff:+.6f} ({r2_pct:+.2f}%)")
    L(f"{'Pearson r':<20} {b1_metrics['pearson_r']:<18.6f} {b2_metrics['pearson_r']:<18.6f} {pearson_diff:+.6f} ({pearson_pct:+.2f}%)")
    L("-" * 70)

    if rmse_diff < 0:
        L(f"RMSE 降低: {abs(rmse_diff):.6f} ({abs(rmse_pct):.2f}% 改善)")
    else:
        L(f"RMSE 升高: {rmse_diff:.6f}")

    if r2_diff > 0:
        L(f"R2 提升: +{r2_diff:.6f}")
    else:
        L(f"R2 下降: {r2_diff:.6f}")

    if pearson_diff > 0:
        L(f"Pearson r 提升: +{pearson_diff:.6f}")
    else:
        L(f"Pearson r 下降: {pearson_diff:.6f}")

    # ============================================================
    # Step 8: Save Results
    # ============================================================
    L("\n" + "=" * 60)
    L("Step 8: Save Results")
    L("=" * 60)

    def make_result(metrics, exp_name, lvl_counts=None):
        result = {
            "experiment_name": exp_name,
            "rmse": round(metrics["rmse"], 6),
            "r2_score": round(metrics["r2_score"], 6),
            "pearson_r": round(metrics["pearson_r"], 6),
            "num_samples_val": int(val_log2.shape[0]),
            "num_proteins": int(len(valid_protein_cols)),
        }
        if lvl_counts is not None:
            result["level_distribution"] = {
                "level_1_strict": lvl_counts.get(1, 0),
                "level_2_strain": lvl_counts.get(2, 0),
                "level_3_global_ctrl": lvl_counts.get(3, 0),
                "level_4_global_mean": lvl_counts.get(4, 0),
            }
        return result

    result_b1 = make_result(b1_metrics, "baseline1_global_mean")
    result_b2 = make_result(b2_metrics, "baseline2_control_backoff", level_counts)

    with open(os.path.join(exp_dir, "result_b1.json"), "w", encoding="utf-8") as f:
        json.dump(result_b1, f, indent=2, ensure_ascii=False)

    with open(os.path.join(exp_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result_b2, f, indent=2, ensure_ascii=False)

    L(f"Baseline 1 result saved to result_b1.json")
    L(f"Baseline 2 result saved to result.json")

    log_file.close()

    # ============================================================
    # Terminal Summary (colored)
    # ============================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"  Directory: {exp_dir}")
    print(f"  Config:    config.yaml")
    print(f"  Log:       log.txt")
    print(f"  Results:   result.json (Baseline 2), result_b1.json (Baseline 1)")
    print("-" * 70)
    print(f"  {'Metric':<20} {'Baseline 1':<15} {'Baseline 2':<15} {'Delta':<15}")
    print("-" * 70)
    print(f"  {'RMSE':<20} {b1_metrics['rmse']:<15.6f} {b2_metrics['rmse']:<15.6f} {rmse_diff:+.6f}")
    print(f"  {'R2 Score':<20} {b1_metrics['r2_score']:<15.6f} {b2_metrics['r2_score']:<15.6f} {r2_diff:+.6f}")
    print(f"  {'Pearson r':<20} {b1_metrics['pearson_r']:<15.6f} {b2_metrics['pearson_r']:<15.6f} {pearson_diff:+.6f}")
    print("-" * 70)
    print(f"  Level Distribution (Baseline 2):")
    print(f"    Level 1 (strict):    {level_counts[1]:>5} samples ({level_counts[1]/n_val*100:.2f}%)")
    print(f"    Level 2 (strain):    {level_counts[2]:>5} samples ({level_counts[2]/n_val*100:.2f}%)")
    print(f"    Level 3 (global ctrl): {level_counts[3]:>5} samples ({level_counts[3]/n_val*100:.2f}%)")
    print(f"    Level 4 (global mean): {level_counts[4]:>5} samples ({level_counts[4]/n_val*100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
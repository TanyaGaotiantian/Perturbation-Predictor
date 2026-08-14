"""
run_transfer.py
L6: 跨数据迁移实验（Cross-dataset Transfer Learning）

设计:
  模拟"外部数据集" → 官方数据集的迁移流程
  - External (pretrain): WAYB + WAYB_rep1 + WAYB_rep2 (不同 batch/source, 模拟外部数据)
  - Internal (finetune): WAYC train (官方训练集)
  - 评估: WAYC val splits (val_seen / val_strain_only / val_chem_only / val_both)

关键控制 (与 scratch 完全一致):
  - 同一 split (WAYC train/val)
  - 同一 representation (Morgan FP + learned strain emb)
  - 同一 residual / loss / lr / epoch (finetune 阶段)
  - 同一 seed
  - 同一模型结构 (C+L+R+P)

Dataset-specific normalization:
  - WAYB 和 WAYC 分别在各自 train 部分做 log2 + per-protein z-score
  - 不直接 concat 原始 abundance
  - 推理时 WAYC val 用 WAYC train 的统计量归一化

对比:
  1. scratch:   从零训练 C+L+R+P on WAYC train
  2. pretrained: WAYB 预训练 → WAYC train finetune (相同 epochs)
"""
import os
import sys
import json
import time
import copy
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, DATASET_DIR)

from chemical_encoder import ChemicalEncoder
from models import AblationModel

# ============ Fixed hyperparameters (与 run_ablation.py 完全一致) ============
FILTER_MISSING_RATE = 0.80
MORGAN_BITS = 1024
EPOCHS = 5
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
MASK_PROB = 0.15
LATENT_DIM = 512
PROTEIN_HIDDEN = 2048
MLP_HIDDEN = 1024
STRAIN_EMB = 32
MEDIUM_EMB = 16
TEMP_EMB = 16
SEED = 42

# 预训练 epochs (在 WAYB 上)
PRETRAIN_EPOCHS = 5


# ============ Data Loading ============

def load_data():
    meta_tv = pd.read_csv(os.path.join(BASE_DIR, "WAYB_WAYC_metadata_train_val(1).csv"))
    meta_te = pd.read_csv(os.path.join(BASE_DIR, "WAYB_WAYC_metadata_test(1).csv"))
    meta = pd.concat([meta_tv, meta_te], ignore_index=True)

    prot_tv = pd.read_csv(os.path.join(BASE_DIR, "WAYB_WAYC_proteome_raw_train_val.csv"))
    prot_te = pd.read_csv(os.path.join(BASE_DIR, "WAYB_WAYC_proteome_raw_test.csv"))
    protein = pd.concat([prot_tv, prot_te], ignore_index=True)

    meta = meta.set_index("sample_ID")
    protein = protein.set_index("sample_ID")
    common = meta.index.intersection(protein.index)
    meta = meta.loc[common]
    protein = protein.loc[common]

    # Filter proteins by missing rate on WAYC train
    wayc_train_mask = (meta["split_final"] == "train") & (meta["data_source"] == "WAYC")
    missing_rate = protein.loc[wayc_train_mask].isna().mean()
    valid_cols = missing_rate[missing_rate < FILTER_MISSING_RATE].index.tolist()
    protein = protein[valid_cols]

    return meta, protein, valid_cols


def build_encoders(meta):
    """Build vocabulary from WAYC train only (official split)."""
    fields = ["Strains", "Medium", "Temperature"]
    encoders = {}
    wayc_train = (meta["split_final"] == "train") & (meta["data_source"] == "WAYC")
    for f in fields:
        train_vals = meta.loc[wayc_train, f].dropna().astype(str).unique().tolist()
        all_vals = ["<UNK>"] + sorted(train_vals)
        encoders[f] = {v: i for i, v in enumerate(all_vals)}
    return encoders


def dataset_specific_normalize(meta, protein, valid_cols):
    """Dataset-specific normalization:
    - WAYB and WAYC are z-scored independently (per-protein, on log2)
    - Statistics computed on each dataset's own train portion
    - Both end up with mean≈0, std≈1 (z-score aligns scales across datasets)
    """
    protein_log2 = np.log2(protein[valid_cols])

    # WAYC train statistics (official)
    wayc_train_mask = (meta["split_final"] == "train") & (meta["data_source"] == "WAYC")
    wayc_train_stats = protein_log2.loc[wayc_train_mask].agg(["mean", "std"])

    # WAYB train statistics (external)
    wayb_train_mask = (meta["split_final"] == "train") & (meta["data_source"].str.startswith("WAYB"))
    wayb_train_stats = protein_log2.loc[wayb_train_mask].agg(["mean", "std"])

    protein_norm = protein_log2.copy()

    # Normalize WAYC samples with WAYC train stats  -> mean≈0, std≈1
    wayc_mask = meta["data_source"] == "WAYC"
    mu_c = wayc_train_stats.loc["mean"].values
    sd_c = wayc_train_stats.loc["std"].values
    sd_c_safe = np.where(sd_c > 1e-6, sd_c, 1.0)
    protein_norm.loc[wayc_mask] = (protein_log2.loc[wayc_mask] - mu_c) / sd_c_safe

    # Normalize WAYB samples with WAYB train stats  -> mean≈0, std≈1
    # (independent z-score; both datasets now share the same target distribution)
    wayb_mask = meta["data_source"].str.startswith("WAYB")
    mu_b = wayb_train_stats.loc["mean"].values
    sd_b = wayb_train_stats.loc["std"].values
    sd_b_safe = np.where(sd_b > 1e-6, sd_b, 1.0)
    protein_norm.loc[wayb_mask] = (protein_log2.loc[wayb_mask] - mu_b) / sd_b_safe

    return protein_norm, wayc_train_stats


# ============ Dataset ============

class TransferDataset(Dataset):
    def __init__(self, meta, protein_norm, sample_ids, encoders, chem_encoder, mask_prob=0.0):
        self.meta = meta.loc[sample_ids].copy()
        self.protein_norm = protein_norm.loc[sample_ids]
        self.sample_ids = list(sample_ids)
        self.encoders = encoders
        self.chem_encoder = chem_encoder
        self.num_proteins = self.protein_norm.shape[1]
        self.mask_prob = mask_prob
        self.chem_dim = chem_encoder.output_dim

    def __len__(self):
        return len(self.sample_ids)

    def _encode(self, field, value):
        v = str(value) if pd.notna(value) else "<UNK>"
        return self.encoders[field].get(v, 0)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        row = self.meta.loc[sid]
        y_row = self.protein_norm.loc[sid].values.astype(np.float32)

        obs = ~np.isnan(y_row)
        y_clean = y_row.copy()
        y_clean[~obs] = 0.0

        x = y_clean.copy()
        x_mask = obs.astype(np.float32)

        if self.mask_prob > 0:
            cand = np.where(obs)[0]
            if len(cand) > 0:
                n = int(np.ceil(len(cand) * self.mask_prob))
                if n > 0:
                    mi = np.random.choice(cand, n, replace=False)
                    x[mi] = 0.0
                    x_mask[mi] = 0.0

        chem_fp = self.chem_encoder.encode(row["perturbation_no_concentration"])

        return {
            "x": torch.from_numpy(x),
            "x_mask": torch.from_numpy(x_mask),
            "y": torch.from_numpy(y_clean),
            "y_mask": torch.from_numpy(obs.astype(np.float32)),
            "strain": torch.tensor(self._encode("Strains", row["Strains"]), dtype=torch.long),
            "medium": torch.tensor(self._encode("Medium", row["Medium"]), dtype=torch.long),
            "temp": torch.tensor(self._encode("Temperature", row["Temperature"]), dtype=torch.long),
            "chem": torch.from_numpy(chem_fp),
        }


# ============ Metrics ============

def compute_metrics(yt_list, yp_list, ym_list):
    yt = np.concatenate(yt_list)
    yp = np.concatenate(yp_list)
    ym = np.concatenate(ym_list).astype(bool)
    yt_v, yp_v = yt[ym], yp[ym]
    if len(yt_v) == 0:
        return {"rmse": float("nan"), "r2": float("nan"), "mae": float("nan"),
                "pearson_r": float("nan"), "n_samples": 0}
    rmse = float(np.sqrt(np.mean((yt_v - yp_v) ** 2)))
    r2 = float(r2_score(yt_v, yp_v))
    mae = float(mean_absolute_error(yt_v, yp_v))
    pr, _ = pearsonr(yt_v, yp_v)
    return {"rmse": rmse, "r2": r2, "mae": mae, "pearson_r": float(pr),
            "n_samples": int(ym.sum())}


def evaluate(model, loader, device):
    model.eval()
    yt, yp, ym = [], [], []
    with torch.no_grad():
        for b in loader:
            preds = model(
                b["x"].to(device), b["x_mask"].to(device),
                b["strain"].to(device), b["medium"].to(device),
                b["temp"].to(device), b["chem"].to(device),
            )
            yt.append(b["y"].cpu().numpy())
            yp.append(preds.cpu().numpy())
            ym.append(b["y_mask"].cpu().numpy())
    return compute_metrics(yt, yp, ym)


# ============ Training ============

def train_epoch(model, loader, optimizer, criterion, device, ep, name=""):
    model.train()
    losses = []
    for bi, b in enumerate(loader):
        x = b["x"].to(device)
        xm = b["x_mask"].to(device)
        y = b["y"].to(device)
        ym = b["y_mask"].to(device)
        strain = b["strain"].to(device)
        medium = b["medium"].to(device)
        temp = b["temp"].to(device)
        chem = b["chem"].to(device)

        optimizer.zero_grad()
        preds = model(x, xm, strain, medium, temp, chem)
        loss_mat = criterion(preds, y)
        loss = (loss_mat * ym).sum() / ym.sum().clamp(min=1e-6)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item())

        if (bi + 1) % 40 == 0:
            print(f"  [{name}] ep{ep} [{bi+1}/{len(loader)}] loss={np.mean(losses[-40:]):.4f}")
    return float(np.mean(losses))


def train_model(model, train_loader, val_loader, device, name="", epochs=EPOCHS, lr=LR):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    print(f"\n{'='*60}")
    print(f"Training: {name}  (epochs={epochs}, lr={lr})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,} ({n_params/1e6:.2f}M)")
    print(f"{'='*60}")

    t0 = time.time()
    history = []
    for ep in range(1, epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device, ep, name)
        m = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        history.append({"epoch": ep, "train_loss": avg_loss, **m})
        print(f"  [{name}] Ep {ep}/{epochs} | loss={avg_loss:.4f} | "
              f"RMSE={m['rmse']:.4f} R2={m['r2']:.4f} MAE={m['mae']:.4f} "
              f"Pearson={m['pearson_r']:.4f} | {elapsed:.0f}s")

    final = evaluate(model, val_loader, device)
    final["train_time_sec"] = round(time.time() - t0, 2)
    final["history"] = history
    final["num_params"] = n_params
    return final


# ============ Main ============

def main():
    print("[Start] L6: Cross-dataset Transfer Learning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    meta, protein, valid_cols = load_data()
    num_proteins = len(valid_cols)
    encoders = build_encoders(meta)
    chem_encoder = ChemicalEncoder(
        smiles_map_path=os.path.join(DATASET_DIR, "smiles_map.json"),
        n_bits=MORGAN_BITS, radius=2
    )

    num_strains = len(encoders["Strains"])
    num_medium = len(encoders["Medium"])
    num_temp = len(encoders["Temperature"])
    chem_dim = chem_encoder.output_dim

    # Dataset-specific normalization
    protein_norm, wayc_stats = dataset_specific_normalize(meta, protein, valid_cols)

    # ===== Define sample splits =====
    # External (pretrain): WAYB train samples (all replicates)
    ext_mask = (meta["split_final"] == "train") & (meta["data_source"].str.startswith("WAYB"))
    ext_ids = meta.loc[ext_mask].index.tolist()

    # Internal (finetune): WAYC train samples
    int_mask = (meta["split_final"] == "train") & (meta["data_source"] == "WAYC")
    int_ids = meta.loc[int_mask].index.tolist()

    # Val: WAYC val splits (official val)
    val_mask = meta["split_final"].str.startswith("val") & (meta["data_source"] == "WAYC")
    val_ids = meta.loc[val_mask].index.tolist()

    # Per-split val ids
    split_val_ids = {}
    for sn in ["val_both", "val_strain_only", "val_chem_only", "val_time"]:
        m = (meta["split_final"] == sn) & (meta["data_source"] == "WAYC")
        split_val_ids[sn] = meta.loc[m].index.tolist()

    print(f"\nData split:")
    print(f"  External (WAYB train, pretrain): {len(ext_ids)} samples")
    print(f"  Internal (WAYC train, finetune): {len(int_ids)} samples")
    print(f"  Val (WAYC val, all):             {len(val_ids)} samples")
    for sn, ids in split_val_ids.items():
        print(f"    {sn}: {len(ids)} samples")

    print(f"\nProteins: {num_proteins}, strains: {num_strains}, chem_dim: {chem_dim}")

    # ===== Build datasets =====
    ext_ds = TransferDataset(meta, protein_norm, ext_ids, encoders, chem_encoder, MASK_PROB)
    int_ds = TransferDataset(meta, protein_norm, int_ids, encoders, chem_encoder, MASK_PROB)
    val_ds = TransferDataset(meta, protein_norm, val_ids, encoders, chem_encoder, 0.0)

    ext_loader = DataLoader(ext_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    int_loader = DataLoader(int_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE * 2, shuffle=False, num_workers=0)

    common_kwargs = dict(
        num_proteins=num_proteins, num_strains=num_strains,
        num_medium=num_medium, num_temp=num_temp, chem_dim=chem_dim,
        strain_emb=STRAIN_EMB, medium_emb=MEDIUM_EMB, temp_emb=TEMP_EMB,
        protein_hidden=PROTEIN_HIDDEN, latent_dim=LATENT_DIM,
        mlp_hidden=MLP_HIDDEN, dropout=0.1,
    )

    results = {}

    # ===== 1. Scratch: train C+L+R+P from zero on WAYC train =====
    print("\n" + "#" * 60)
    print("# Experiment 1: SCRATCH (train from zero on WAYC train)")
    print("#" * 60)
    torch.manual_seed(SEED)
    model_scratch = AblationModel(**common_kwargs, use_residual=True, use_latent=True,
                                   use_cross=True, use_prior=True).to(device)
    results["scratch"] = train_model(model_scratch, int_loader, val_loader, device,
                                      name="scratch", epochs=EPOCHS, lr=LR)

    # ===== 2. Pretrained: pretrain on WAYB → finetune on WAYC train =====
    print("\n" + "#" * 60)
    print("# Experiment 2: PRETRAINED (WAYB pretrain → WAYC finetune)")
    print("#" * 60)

    # 2a. Pretrain on WAYB
    torch.manual_seed(SEED)
    model_pre = AblationModel(**common_kwargs, use_residual=True, use_latent=True,
                               use_cross=True, use_prior=True).to(device)
    print("\n--- Stage 1: Pretrain on WAYB (external) ---")
    # Use WAYC val as a sanity-check monitor during pretraining (not for selection)
    pretrain_result = train_model(model_pre, ext_loader, val_loader, device,
                                   name="pretrain", epochs=PRETRAIN_EPOCHS, lr=LR)

    # Save pretrained weights snapshot
    pretrained_state = copy.deepcopy(model_pre.state_dict())
    pretrain_ckpt = os.path.join(BASE_DIR, "transfer_pretrained_weights.pt")
    torch.save(pretrained_state, pretrain_ckpt)
    print(f"\nPretrained weights saved to {pretrain_ckpt}")

    # 2b. Finetune on WAYC train (same epochs as scratch)
    print("\n--- Stage 2: Finetune on WAYC train (internal) ---")
    # Continue training the same model on WAYC train
    finetune_result = train_model(model_pre, int_loader, val_loader, device,
                                   name="finetune", epochs=EPOCHS, lr=LR)
    results["pretrained"] = finetune_result
    results["pretrained"]["pretrain_history"] = pretrain_result.get("history", [])
    results["pretrained"]["pretrain_final_metrics"] = {k: v for k, v in pretrain_result.items()
                                                        if k not in ("history",)}

    # ===== Per-split evaluation for both models =====
    print("\n" + "=" * 60)
    print("Per-split Evaluation (WAYC val splits)")
    print("=" * 60)

    split_eval = {"scratch": {}, "pretrained": {}}
    for sn, ids in split_val_ids.items():
        if len(ids) == 0:
            continue
        ds = TransferDataset(meta, protein_norm, ids, encoders, chem_encoder, 0.0)
        loader = DataLoader(ds, BATCH_SIZE * 2, shuffle=False, num_workers=0)

        m_s = evaluate(model_scratch, loader, device)
        m_p = evaluate(model_pre, loader, device)
        split_eval["scratch"][sn] = m_s
        split_eval["pretrained"][sn] = m_p

        delta_rmse = m_s["rmse"] - m_p["rmse"]
        arrow = "↑" if delta_rmse > 0 else "↓"
        print(f"  {sn:<20} n={len(ids):<5} | "
              f"scratch RMSE={m_s['rmse']:.4f} R2={m_s['r2']:.4f} | "
              f"pretrained RMSE={m_p['rmse']:.4f} R2={m_p['r2']:.4f} | "
              f"ΔRMSE={delta_rmse:+.4f} ({arrow}pretrained better)")

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("L6 TRANSFER LEARNING SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<15} {'RMSE':<10} {'R2':<10} {'MAE':<10} {'Pearson':<10}")
    print("-" * 55)
    for name in ["scratch", "pretrained"]:
        r = results[name]
        print(f"{name:<15} {r['rmse']:<10.4f} {r['r2']:<10.4f} {r['mae']:<10.4f} "
              f"{r['pearson_r']:<10.4f}")

    print("\nPer-split (RMSE / R2):")
    print(f"{'Split':<20} {'scratch':<22} {'pretrained':<22} {'ΔRMSE':<10}")
    print("-" * 74)
    for sn in split_val_ids:
        if sn not in split_eval["scratch"]:
            continue
        s = split_eval["scratch"][sn]
        p = split_eval["pretrained"][sn]
        delta = s["rmse"] - p["rmse"]
        print(f"{sn:<20} {s['rmse']:.4f} / {s['r2']:.4f}    {p['rmse']:.4f} / {p['r2']:.4f}    {delta:+.4f}")

    # ===== Save report =====
    report = {
        "fixed_config": {
            "epochs_finetune": EPOCHS,
            "epochs_pretrain": PRETRAIN_EPOCHS,
            "lr": LR, "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY, "mask_prob": MASK_PROB,
            "loss": "MSELoss(masked)", "seed": SEED,
            "model": "C+L+R+P (Cross-Attn + Latent + Residual + ProteinPrior)",
            "external_source": "WAYB + WAYB_rep1 + WAYB_rep2 (train)",
            "internal_source": "WAYC (train)",
            "eval_source": "WAYC (val splits)",
            "normalization": "dataset-specific log2 z-score (WAYB/WAYC separate), aligned to WAYC mean",
        },
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                     for k, v in results.items()},
        "per_split_eval": split_eval,
    }

    out_path = os.path.join(BASE_DIR, "transfer_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {out_path}")

    # Save history
    history_data = {
        "scratch": results["scratch"].get("history", []),
        "pretrained_pretrain": results["pretrained"].get("pretrain_history", []),
        "pretrained_finetune": [h for h in results["pretrained"].get("history", [])
                                 if isinstance(h, dict)],
    }
    hist_path = os.path.join(BASE_DIR, "transfer_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)
    print(f"History saved to {hist_path}")


if __name__ == "__main__":
    main()

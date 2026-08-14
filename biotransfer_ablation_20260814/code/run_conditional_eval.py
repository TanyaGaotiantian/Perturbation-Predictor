"""
run_conditional_eval.py
L7: 深度条件级评估

对最佳模型 (C+L+R+P) 做:
  1. 分 split 评估: val_seen / val_strain_only / val_chem_only / val_both / val_time
  2. 条件级分析: 按 strain / compound / time / temperature 分别算 RMSE
  3. Protein 级解释: 哪些 protein 预测最准/最差
  4. 模块贡献: 去掉 C / 去掉 P 对 val_both 的影响

控制变量: 与 run_ablation.py 完全一致 (同 split/loss/lr/epoch/seed)
"""
import os
import sys
import json
import time
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

# ============ Fixed hyperparameters (与 run_ablation.py 一致) ============
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


# ============ Data ============

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

    train_mask = meta["split_final"] == "train"
    missing_rate = protein.loc[train_mask].isna().mean()
    valid_cols = missing_rate[missing_rate < FILTER_MISSING_RATE].index.tolist()
    protein = protein[valid_cols]
    protein_log2 = np.log2(protein)

    return meta, protein_log2, valid_cols


def build_encoders(meta):
    fields = ["Strains", "Medium", "Temperature"]
    encoders = {}
    for f in fields:
        train_vals = meta.loc[meta["split_final"] == "train", f].dropna().astype(str).unique().tolist()
        all_vals = ["<UNK>"] + sorted(train_vals)
        encoders[f] = {v: i for i, v in enumerate(all_vals)}
    return encoders


class EvalDataset(Dataset):
    def __init__(self, meta, protein_log2, split_name, encoders, chem_encoder, mask_prob=0.0,
                 sample_ids=None):
        if sample_ids is not None:
            self.meta = meta.loc[sample_ids].copy()
        elif split_name == "train":
            self.meta = meta.loc[meta["split_final"] == "train"].copy()
        elif split_name == "val":
            self.meta = meta.loc[meta["split_final"].str.startswith("val")].copy()
        else:
            self.meta = meta.loc[meta["split_final"] == split_name].copy()

        self.protein_log2 = protein_log2.loc[self.meta.index]
        self.sample_ids = self.meta.index.tolist()
        self.encoders = encoders
        self.chem_encoder = chem_encoder
        self.num_proteins = self.protein_log2.shape[1]
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
        y_row = self.protein_log2.loc[sid].values.astype(np.float32)

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
            "sample_id": sid,
        }


# ============ Metrics ============

def compute_metrics(yt, yp, ym):
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


def predict_all(model, loader, device):
    """Return per-sample predictions and ground truth + masks + metadata."""
    model.eval()
    all_preds, all_y, all_ym = [], [], []
    all_meta_rows = []
    with torch.no_grad():
        for b in loader:
            preds = model(
                b["x"].to(device), b["x_mask"].to(device),
                b["strain"].to(device), b["medium"].to(device),
                b["temp"].to(device), b["chem"].to(device),
            )
            all_preds.append(preds.cpu().numpy())
            all_y.append(b["y"].numpy())
            all_ym.append(b["y_mask"].numpy())
            for sid in b["sample_id"]:
                all_meta_rows.append(sid)

    preds = np.concatenate(all_preds)   # (N, P)
    y = np.concatenate(all_y)            # (N, P)
    ym = np.concatenate(all_ym).astype(bool)  # (N, P)
    return preds, y, ym, all_meta_rows


# ============ Training ============

def train_model(model, train_loader, val_loader, device, name=""):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"\nTraining: {name}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}")

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bi, b in enumerate(train_loader):
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

            if (bi + 1) % 60 == 0:
                print(f"  ep{ep} [{bi+1}/{len(train_loader)}] loss={np.mean(losses[-60:]):.4f}")

        model.eval()
        yt, yp, ym_l = [], [], []
        with torch.no_grad():
            for b in val_loader:
                preds = model(b["x"].to(device), b["x_mask"].to(device),
                              b["strain"].to(device), b["medium"].to(device),
                              b["temp"].to(device), b["chem"].to(device))
                yt.append(b["y"].numpy())
                yp.append(preds.cpu().numpy())
                ym_l.append(b["y_mask"].numpy())
        m = compute_metrics(np.concatenate(yt), np.concatenate(yp),
                            np.concatenate(ym_l).astype(bool))
        print(f"  Ep {ep}/{EPOCHS} | loss={np.mean(losses):.4f} | "
              f"RMSE={m['rmse']:.4f} R2={m['r2']:.4f} | {time.time()-t0:.0f}s")

    return model


# ============ Conditional Analysis ============

def conditional_analysis(preds, y, ym, sample_ids, meta, group_field):
    """Group samples by a metadata field and compute per-group metrics."""
    results = {}
    sid_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

    groups = meta.loc[sample_ids, group_field].astype(str)
    for gval in sorted(groups.unique()):
        g_sids = groups[groups == gval].index.tolist()
        idxs = [sid_to_idx[s] for s in g_sids]
        if len(idxs) == 0:
            continue
        p_g = preds[idxs]
        y_g = y[idxs]
        m_g = ym[idxs]
        m = compute_metrics(y_g, p_g, m_g)
        m["n_samples_in_group"] = len(idxs)
        results[gval] = m

    return results


def protein_level_analysis(preds, y, ym, protein_cols, top_k=15):
    """Per-protein RMSE: which proteins are best/worst predicted."""
    n_proteins = preds.shape[1]
    per_protein = []
    for j in range(n_proteins):
        mask_j = ym[:, j]
        if mask_j.sum() < 50:
            continue
        yt_j = y[mask_j, j]
        yp_j = preds[mask_j, j]
        rmse = float(np.sqrt(np.mean((yt_j - yp_j) ** 2)))
        mae = float(np.mean(np.abs(yt_j - yp_j)))
        pr, _ = pearsonr(yt_j, yp_j) if len(yt_j) > 2 else (float("nan"), 0)
        per_protein.append({
            "protein": protein_cols[j],
            "idx": j,
            "rmse": rmse,
            "mae": mae,
            "pearson_r": float(pr),
            "n_obs": int(mask_j.sum()),
        })

    df = pd.DataFrame(per_protein).sort_values("rmse")
    best = df.head(top_k).to_dict(orient="records")
    worst = df.tail(top_k).iloc[::-1].to_dict(orient="records")
    stats = {
        "mean_rmse": float(df["rmse"].mean()),
        "median_rmse": float(df["rmse"].median()),
        "std_rmse": float(df["rmse"].std()),
        "mean_pearson": float(df["pearson_r"].mean()),
        "n_proteins_evaluated": len(df),
    }
    return {"best": best, "worst": worst, "stats": stats, "all": df.to_dict(orient="records")}


# ============ Main ============

def main():
    print("[Start] L7: Conditional Evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    meta, protein_log2, valid_cols = load_data()
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
    print(f"Proteins: {num_proteins}, strains: {num_strains}, chem_dim: {chem_dim}")

    train_ds = EvalDataset(meta, protein_log2, "train", encoders, chem_encoder, MASK_PROB)
    val_ds = EvalDataset(meta, protein_log2, "val", encoders, chem_encoder, 0.0)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE * 2, shuffle=False, num_workers=0)

    common_kwargs = dict(
        num_proteins=num_proteins, num_strains=num_strains,
        num_medium=num_medium, num_temp=num_temp, chem_dim=chem_dim,
        strain_emb=STRAIN_EMB, medium_emb=MEDIUM_EMB, temp_emb=TEMP_EMB,
        protein_hidden=PROTEIN_HIDDEN, latent_dim=LATENT_DIM,
        mlp_hidden=MLP_HIDDEN, dropout=0.1,
    )

    # ===== Train full model (C+L+R+P) =====
    print("\n" + "=" * 60)
    print("Train C+L+R+P (full model) for conditional evaluation")
    print("=" * 60)
    torch.manual_seed(SEED)
    model_full = AblationModel(**common_kwargs, use_residual=True, use_latent=True,
                                use_cross=True, use_prior=True).to(device)
    model_full = train_model(model_full, train_loader, val_loader, device, "C+L+R+P")

    # Save full model
    ckpt_full = os.path.join(BASE_DIR, "cond_eval_model_CLRP.pt")
    torch.save(model_full.state_dict(), ckpt_full)
    print(f"Saved: {ckpt_full}")

    # ===== Train ablation variants for module contribution =====
    print("\n" + "=" * 60)
    print("Train ablation variants for module-contribution analysis")
    print("=" * 60)

    # C+L+R (without P)
    torch.manual_seed(SEED)
    model_noP = AblationModel(**common_kwargs, use_residual=True, use_latent=True,
                               use_cross=True, use_prior=False).to(device)
    model_noP = train_model(model_noP, train_loader, val_loader, device, "C+L+R (no P)")

    # L+R (without C, without P)
    torch.manual_seed(SEED)
    model_noC = AblationModel(**common_kwargs, use_residual=True, use_latent=True,
                               use_cross=False, use_prior=False).to(device)
    model_noC = train_model(model_noC, train_loader, val_loader, device, "L+R (no C, no P)")

    # R only
    torch.manual_seed(SEED)
    model_R = AblationModel(**common_kwargs, use_residual=True, use_latent=False,
                             use_cross=False, use_prior=False).to(device)
    model_R = train_model(model_R, train_loader, val_loader, device, "R only")

    # ===== 1. Per-split evaluation =====
    print("\n" + "=" * 60)
    print("1. Per-split Evaluation (all models)")
    print("=" * 60)

    split_names = ["val_both", "val_strain_only", "val_chem_only", "val_time"]
    per_split = {}
    models = {"R": model_R, "L+R": model_noC, "C+L+R": model_noP, "C+L+R+P": model_full}

    # Also compute val_seen (strain_role=train & chemical_role=train within val)
    val_seen_mask = (meta["split_final"].str.startswith("val") &
                     (meta["strain_role"] == "train") &
                     (meta["chemical_role"] == "train"))
    val_seen_ids = meta.loc[val_seen_mask].index.tolist()

    for mname, mdl in models.items():
        per_split[mname] = {}
        for sn in split_names:
            ids = meta.loc[meta["split_final"] == sn].index.tolist()
            if len(ids) == 0:
                continue
            ds = EvalDataset(meta, protein_log2, None, encoders, chem_encoder, 0.0, sample_ids=ids)
            loader = DataLoader(ds, BATCH_SIZE * 2, shuffle=False, num_workers=0)
            preds, y, ym, _ = predict_all(mdl, loader, device)
            m = compute_metrics(y, preds, ym)
            per_split[mname][sn] = m
            print(f"  [{mname:<10}] {sn:<20} n={len(ids):<5} RMSE={m['rmse']:.4f} R2={m['r2']:.4f}")

        # val_seen
        if len(val_seen_ids) > 0:
            ds = EvalDataset(meta, protein_log2, None, encoders, chem_encoder, 0.0, sample_ids=val_seen_ids)
            loader = DataLoader(ds, BATCH_SIZE * 2, shuffle=False, num_workers=0)
            preds, y, ym, _ = predict_all(mdl, loader, device)
            m = compute_metrics(y, preds, ym)
            per_split[mname]["val_seen"] = m
            print(f"  [{mname:<10}] {'val_seen':<20} n={len(val_seen_ids):<5} RMSE={m['rmse']:.4f} R2={m['r2']:.4f}")

    # ===== 2. Conditional analysis on full model =====
    print("\n" + "=" * 60)
    print("2. Conditional Analysis (C+L+R+P model, on val)")
    print("=" * 60)

    preds_val, y_val, ym_val, sids_val = predict_all(model_full, val_loader, device)

    conditional = {}
    for field in ["Strains", "perturbation_no_concentration", "pert_time", "Temperature", "Medium"]:
        res = conditional_analysis(preds_val, y_val, ym_val, sids_val, meta, field)
        conditional[field] = res
        print(f"\n  --- By {field} ---")
        sorted_res = sorted(res.items(), key=lambda x: x[1]["rmse"])
        for gval, m in sorted_res:
            print(f"    {str(gval)[:30]:<32} n={m['n_samples_in_group']:<5} "
                  f"RMSE={m['rmse']:.4f} R2={m['r2']:.4f}")

    # ===== 3. Protein-level analysis =====
    print("\n" + "=" * 60)
    print("3. Protein-level Analysis (C+L+R+P model, on val)")
    print("=" * 60)

    protein_analysis = protein_level_analysis(preds_val, y_val, ym_val, valid_cols, top_k=15)
    print(f"\n  Proteins evaluated: {protein_analysis['stats']['n_proteins_evaluated']}")
    print(f"  Mean per-protein RMSE: {protein_analysis['stats']['mean_rmse']:.4f}")
    print(f"  Median per-protein RMSE: {protein_analysis['stats']['median_rmse']:.4f}")
    print(f"  Mean per-protein Pearson: {protein_analysis['stats']['mean_pearson']:.4f}")

    print(f"\n  Top-15 BEST predicted proteins (lowest RMSE):")
    print(f"    {'Protein':<20} {'RMSE':<10} {'MAE':<10} {'Pearson':<10} {'N_obs':<8}")
    for r in protein_analysis["best"]:
        print(f"    {r['protein'][:18]:<20} {r['rmse']:<10.4f} {r['mae']:<10.4f} "
              f"{r['pearson_r']:<10.4f} {r['n_obs']:<8}")

    print(f"\n  Top-15 WORST predicted proteins (highest RMSE):")
    print(f"    {'Protein':<20} {'RMSE':<10} {'MAE':<10} {'Pearson':<10} {'N_obs':<8}")
    for r in protein_analysis["worst"]:
        print(f"    {r['protein'][:18]:<20} {r['rmse']:<10.4f} {r['mae']:<10.4f} "
              f"{r['pearson_r']:<10.4f} {r['n_obs']:<8}")

    # ===== 4. Module contribution on val_both (OOD) =====
    print("\n" + "=" * 60)
    print("4. Module Contribution on val_both (OOD: unseen strain + chemical)")
    print("=" * 60)

    both_ids = meta.loc[meta["split_final"] == "val_both"].index.tolist()
    if len(both_ids) > 0:
        ds_both = EvalDataset(meta, protein_log2, None, encoders, chem_encoder, 0.0, sample_ids=both_ids)
        loader_both = DataLoader(ds_both, BATCH_SIZE * 2, shuffle=False, num_workers=0)

        module_contrib = {}
        for mname, mdl in models.items():
            preds, y, ym, _ = predict_all(mdl, loader_both, device)
            m = compute_metrics(y, preds, ym)
            module_contrib[mname] = m
            print(f"  {mname:<12} RMSE={m['rmse']:.4f} R2={m['r2']:.4f} MAE={m['mae']:.4f} "
                  f"Pearson={m['pearson_r']:.4f}")

        # Deltas relative to full
        print(f"\n  Deltas vs C+L+R+P (positive = module helps):")
        base = module_contrib["C+L+R+P"]["rmse"]
        for mname in ["R", "L+R", "C+L+R"]:
            delta = module_contrib[mname]["rmse"] - base
            print(f"    Remove → {mname:<10}: ΔRMSE = {delta:+.4f} "
                  f"({'worse without full' if delta > 0 else 'better without full'})")

    # ===== Save report =====
    report = {
        "fixed_config": {
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY, "mask_prob": MASK_PROB,
            "loss": "MSELoss(masked)", "seed": SEED,
            "model_variants": ["R", "L+R", "C+L+R", "C+L+R+P"],
        },
        "per_split_eval": per_split,
        "conditional_analysis": conditional,
        "protein_level": {
            "stats": protein_analysis["stats"],
            "best_top15": protein_analysis["best"],
            "worst_top15": protein_analysis["worst"],
        },
        "module_contribution_val_both": module_contrib if len(both_ids) > 0 else {},
    }

    out_path = os.path.join(BASE_DIR, "conditional_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()

"""
run_ablation.py
统一消融实验脚本

固定变量:
  - split: train / val (含子split)
  - representation: Morgan FP (chemical) + learned embedding (strain)
  - loss: MSELoss (masked)
  - lr: 3e-4
  - epochs: 5
  - batch_size: 64
  - seed: 42

实验列表:
  L4 残差对比:
    1. MLP           — base
    2. MLP+R         — + residual

  L5 交互模块:
    3. Hadamard      — strain ⊙ drug
    4. InteractMLP   — interaction MLP

  L5 消融:
    5. MLP           — base
    6. MLP+R         — R
    7. MLP+L+R       — L+R
    8. C+L+R         — C+L+R
    9. C+L+R+P       — C+L+R+P

  L7 分split评估:
    对最佳模型按 val_seen / val_strain_only / val_chem_only / val_both 评估
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
from models import AblationModel, InteractionHadamard, InteractionMLP

# ============ Fixed hyperparameters ============
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


# ============ Dataset ============

class AblationDataset(Dataset):
    def __init__(self, meta, protein_log2, split_name, encoders, chem_encoder, mask_prob=0.0):
        if split_name == "train":
            mask = meta["split_final"] == "train"
        elif split_name == "val":
            mask = meta["split_final"].str.startswith("val")
        else:
            mask = meta["split_final"] == split_name

        self.meta = meta.loc[mask].copy()
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
        }


# ============ Metrics ============

def compute_metrics(yt_list, yp_list, ym_list):
    yt = np.concatenate(yt_list)
    yp = np.concatenate(yp_list)
    ym = np.concatenate(ym_list).astype(bool)
    yt_v, yp_v = yt[ym], yp[ym]

    rmse = float(np.sqrt(np.mean((yt_v - yp_v) ** 2)))
    r2 = float(r2_score(yt_v, yp_v))
    mae = float(mean_absolute_error(yt_v, yp_v))
    pr, _ = pearsonr(yt_v, yp_v)
    return {"rmse": rmse, "r2": r2, "mae": mae, "pearson_r": float(pr), "n_samples": int(ym.sum())}


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

def train_model(model, train_loader, val_loader, device, name=""):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,} ({n_params/1e6:.2f}M)")
    print(f"{'='*60}")

    t0 = time.time()
    history = []

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

            if (bi + 1) % 40 == 0:
                print(f"  ep{ep} [{bi+1}/{len(train_loader)}] loss={np.mean(losses[-40:]):.4f}")

        m = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), **m})
        print(f"  Ep {ep}/{EPOCHS} | loss={np.mean(losses):.4f} | "
              f"RMSE={m['rmse']:.4f} R2={m['r2']:.4f} MAE={m['mae']:.4f} "
              f"Pearson={m['pearson_r']:.4f} | {elapsed:.0f}s")

    final = evaluate(model, val_loader, device)
    final["train_time_sec"] = round(time.time() - t0, 2)
    final["history"] = history
    final["num_params"] = n_params
    return final


# ============ Main ============

def main():
    print("[Start] L3-L5-L7 Unified Ablation Experiments")
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

    print(f"\nData: {num_proteins} proteins, {num_strains} strains, {chem_dim} chem dim")
    print(f"Fixed: epochs={EPOCHS}, lr={LR}, bs={BATCH_SIZE}, loss=MSE, mask={MASK_PROB}, seed={SEED}")

    train_ds = AblationDataset(meta, protein_log2, "train", encoders, chem_encoder, MASK_PROB)
    val_ds = AblationDataset(meta, protein_log2, "val", encoders, chem_encoder, 0.0)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE * 2, shuffle=False, num_workers=0)

    common_kwargs = dict(
        num_proteins=num_proteins, num_strains=num_strains,
        num_medium=num_medium, num_temp=num_temp, chem_dim=chem_dim,
        strain_emb=STRAIN_EMB, medium_emb=MEDIUM_EMB, temp_emb=TEMP_EMB,
        protein_hidden=PROTEIN_HIDDEN, latent_dim=LATENT_DIM,
        mlp_hidden=MLP_HIDDEN, dropout=0.1,
    )

    results = {}

    # ===== L4: MLP vs MLP+R =====
    print("\n" + "="*60)
    print("L4: Residual Decomposition Comparison")
    print("="*60)

    model_mlp = AblationModel(**common_kwargs, use_residual=False, use_latent=False,
                              use_cross=False, use_prior=False).to(device)
    results["MLP"] = train_model(model_mlp, train_loader, val_loader, device, "MLP (base)")

    model_mlp_r = AblationModel(**common_kwargs, use_residual=True, use_latent=False,
                                use_cross=False, use_prior=False).to(device)
    results["MLP+R"] = train_model(model_mlp_r, train_loader, val_loader, device, "MLP+R (residual)")

    # ===== L5: Interaction =====
    print("\n" + "="*60)
    print("L5: Interaction Module Comparison")
    print("="*60)

    # Baseline' : MLP (already done above as "MLP")
    results["Baseline_MLP"] = results["MLP"]

    model_hadamard = InteractionHadamard(**common_kwargs).to(device)
    results["Hadamard"] = train_model(model_hadamard, train_loader, val_loader, device, "Hadamard (strain⊙drug)")

    model_interact = InteractionMLP(**common_kwargs).to(device)
    results["InteractMLP"] = train_model(model_interact, train_loader, val_loader, device, "Interaction MLP")

    # ===== L5: Ablation MLP→R→L+R→C+L+R→C+L+R+P =====
    print("\n" + "="*60)
    print("L5: Ablation Study (progressive)")
    print("="*60)

    # MLP (already done)
    results["ablation_MLP"] = results["MLP"]

    # MLP+R (already done)
    results["ablation_R"] = results["MLP+R"]

    # L+R
    model_lr = AblationModel(**common_kwargs, use_residual=True, use_latent=True,
                             use_cross=False, use_prior=False).to(device)
    results["ablation_L+R"] = train_model(model_lr, train_loader, val_loader, device, "L+R")

    # C+L+R
    model_clr = AblationModel(**common_kwargs, use_residual=True, use_latent=True,
                              use_cross=True, use_prior=False).to(device)
    results["ablation_C+L+R"] = train_model(model_clr, train_loader, val_loader, device, "C+L+R")

    # C+L+R+P
    model_clrp = AblationModel(**common_kwargs, use_residual=True, use_latent=True,
                               use_cross=True, use_prior=True).to(device)
    results["ablation_C+L+R+P"] = train_model(model_clrp, train_loader, val_loader, device, "C+L+R+P")

    # ===== L7: Per-split evaluation for best model =====
    print("\n" + "="*60)
    print("L7: Per-split Evaluation (C+L+R+P model)")
    print("="*60)

    split_names = ["val_both", "val_strain_only", "val_chem_only"]
    split_results = {}
    for sn in split_names:
        if sn in meta["split_final"].values:
            ds = AblationDataset(meta, protein_log2, sn, encoders, chem_encoder, 0.0)
            if len(ds) > 0:
                loader = DataLoader(ds, BATCH_SIZE * 2, shuffle=False, num_workers=0)
                m = evaluate(model_clrp, loader, device)
                split_results[sn] = m
                print(f"  {sn}: n={len(ds)} | RMSE={m['rmse']:.4f} R2={m['r2']:.4f} "
                      f"MAE={m['mae']:.4f} Pearson={m['pearson_r']:.4f}")

    # Also evaluate overall val (seen conditions)
    val_seen = meta[meta["split_final"].str.startswith("val") &
                     (meta["strain_role"] == "train") &
                     (meta["chemical_role"] == "train")]
    if len(val_seen) > 0:
        ds = AblationDataset(meta, protein_log2, "val", encoders, chem_encoder, 0.0)
        # Filter to seen
        seen_ids = val_seen.index
        ds.sample_ids = [s for s in ds.sample_ids if s in seen_ids]
        if len(ds) > 0:
            loader = DataLoader(ds, BATCH_SIZE * 2, shuffle=False, num_workers=0)
            m = evaluate(model_clrp, loader, device)
            split_results["val_seen"] = m
            print(f"  val_seen: n={len(ds)} | RMSE={m['rmse']:.4f} R2={m['r2']:.4f} "
                  f"MAE={m['mae']:.4f} Pearson={m['pearson_r']:.4f}")

    # ===== Summary =====
    print("\n" + "="*60)
    print("FULL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Experiment':<25} {'RMSE':<10} {'R2':<10} {'MAE':<10} {'Pearson':<10} {'Params':<12}")
    print("-" * 77)

    for name in ["MLP", "MLP+R", "Hadamard", "InteractMLP",
                 "ablation_MLP", "ablation_R", "ablation_L+R",
                 "ablation_C+L+R", "ablation_C+L+R+P"]:
        r = results[name]
        print(f"{name:<25} {r['rmse']:<10.4f} {r['r2']:<10.4f} {r['mae']:<10.4f} "
              f"{r['pearson_r']:<10.4f} {r['num_params']:<12,}")

    # Save report
    report = {
        "fixed_config": {
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY, "mask_prob": MASK_PROB,
            "loss": "MSELoss(masked)", "seed": SEED,
            "latent_dim": LATENT_DIM, "protein_hidden": PROTEIN_HIDDEN,
            "mlp_hidden": MLP_HIDDEN, "strain_emb": STRAIN_EMB,
            "chem_encoding": "Morgan FP (1024 bits, radius=2)",
            "strain_encoding": "learned embedding (32 dim, transferable)",
        },
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                     for k, v in results.items()},
        "per_split_eval": split_results,
    }

    out_path = os.path.join(BASE_DIR, "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {out_path}")

    # Save history for plotting
    history_data = {}
    for name, r in results.items():
        if "history" in r:
            history_data[name] = r["history"]

    hist_path = os.path.join(BASE_DIR, "ablation_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)
    print(f"History saved to {hist_path}")


if __name__ == "__main__":
    main()

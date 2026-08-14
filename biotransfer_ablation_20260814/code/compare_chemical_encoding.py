"""
compare_chemical_encoding.py
对比实验: one-hot vs Morgan fingerprint 化学品编码

单变量控制:
  - Loss: MSELoss (masked) — 两个实验完全相同
  - 模型结构: protein_mlp + context_mlp -> latent -> output_mlp — 完全相同
  - 训练参数: epochs, lr, batch_size, weight_decay — 完全相同
  - 数据划分: train/val split — 完全相同
  - 掩码: mask_prob=0.15 — 完全相同

唯一变量: chemical 特征编码方式
  A) one-hot:  perturbation_no_concentration -> one-hot (57 dim)
  B) morgan:   perturbation_no_concentration -> SMILES -> Morgan FP (1024 dim)
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
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, DATASET_DIR)

from chemical_encoder import ChemicalEncoder

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
SEED = 42


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

    # Filter proteins by missing rate on train
    train_mask = meta["split_final"] == "train"
    missing_rate = protein.loc[train_mask].isna().mean()
    valid_cols = missing_rate[missing_rate < FILTER_MISSING_RATE].index.tolist()
    protein = protein[valid_cols]
    protein_log2 = np.log2(protein)

    return meta, protein_log2, valid_cols


def build_cat_encoders(meta):
    """Build vocabulary for categorical fields (train-only values + <UNK>)."""
    fields = ["Strains", "Medium", "Temperature", "perturbation_no_concentration"]
    encoders = {}
    for f in fields:
        train_vals = meta.loc[meta["split_final"] == "train", f].dropna().astype(str).unique().tolist()
        all_vals = ["<UNK>"] + sorted(train_vals)
        encoders[f] = {v: i for i, v in enumerate(all_vals)}
    return encoders


# ============ Dataset ============

class ChemDataset(Dataset):
    """
    通用 Dataset，支持两种 chemical 编码模式。

    Args:
        chem_mode: "one_hot" or "morgan"
    """

    def __init__(self, meta, protein_log2, split_name, encoders,
                 chem_mode="one_hot", chem_encoder=None,
                 mask_prob=0.0):
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
        self.num_proteins = self.protein_log2.shape[1]
        self.mask_prob = mask_prob
        self.chem_mode = chem_mode
        self.chem_encoder = chem_encoder

        self.cat_fields = ["Strains", "Medium", "Temperature"]
        self.num_strains = len(encoders["Strains"])
        self.num_medium = len(encoders["Medium"])
        self.num_temp = len(encoders["Temperature"])
        self.num_chems = len(encoders["perturbation_no_concentration"])

    @property
    def context_dim(self):
        """Non-protein context feature dimension."""
        dim = self.num_strains + self.num_medium + self.num_temp
        if self.chem_mode == "one_hot":
            dim += self.num_chems
        else:
            dim += self.chem_encoder.output_dim
        return dim

    def _encode_cat(self, field, value):
        v = str(value) if pd.notna(value) else "<UNK>"
        return self.encoders[field].get(v, 0)

    def _one_hot(self, idx, size):
        v = np.zeros(size, dtype=np.float32)
        v[idx] = 1.0
        return v

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        row = self.meta.loc[sid]
        y_row = self.protein_log2.loc[sid].values.astype(np.float32)

        obs_mask = ~np.isnan(y_row)
        y_clean = y_row.copy()
        y_clean[~obs_mask] = 0.0

        input_vec = y_clean.copy()
        input_mask = obs_mask.astype(np.float32)

        if self.mask_prob > 0.0:
            cand = np.where(obs_mask)[0]
            if len(cand) > 0:
                n_mask = int(np.ceil(len(cand) * self.mask_prob))
                if n_mask > 0:
                    mask_idx = np.random.choice(cand, size=n_mask, replace=False)
                    input_vec[mask_idx] = 0.0
                    input_mask[mask_idx] = 0.0

        # Context features
        s_oh = self._one_hot(self._encode_cat("Strains", row["Strains"]), self.num_strains)
        m_oh = self._one_hot(self._encode_cat("Medium", row["Medium"]), self.num_medium)
        t_oh = self._one_hot(self._encode_cat("Temperature", row["Temperature"]), self.num_temp)

        if self.chem_mode == "one_hot":
            c_oh = self._one_hot(
                self._encode_cat("perturbation_no_concentration", row["perturbation_no_concentration"]),
                self.num_chems
            )
        else:
            c_oh = self.chem_encoder.encode(row["perturbation_no_concentration"])

        context = np.concatenate([s_oh, m_oh, t_oh, c_oh], axis=0).astype(np.float32)

        return {
            "x": torch.from_numpy(input_vec),
            "x_mask": torch.from_numpy(input_mask),
            "y": torch.from_numpy(y_clean),
            "y_mask": torch.from_numpy(obs_mask.astype(np.float32)),
            "context": torch.from_numpy(context),
        }


# ============ Model (identical for both experiments) ============

class ChemPredictor(nn.Module):
    """
    protein + context -> latent -> protein prediction

    Structure is IDENTICAL for both experiments.
    Only context_dim changes based on chem encoding.
    """

    def __init__(self, num_proteins, context_dim,
                 protein_hidden=2048, latent_dim=512, mlp_hidden=1024, dropout=0.1):
        super().__init__()
        self.num_proteins = num_proteins
        self.latent_dim = latent_dim

        # Protein encoder: [x, x_mask] -> latent
        self.protein_mlp = nn.Sequential(
            nn.Linear(num_proteins * 2, protein_hidden),
            nn.LayerNorm(protein_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(protein_hidden, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        # Context encoder: context_dim -> latent_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion -> output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_proteins),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_mask, context):
        p_in = torch.cat([x, x_mask], dim=-1)
        p_z = self.protein_mlp(p_in)
        c_z = self.context_mlp(context)
        z = torch.cat([p_z, c_z], dim=-1)
        return self.decoder(z)


# ============ Training & Evaluation ============

def compute_metrics(y_true_list, y_pred_list, y_mask_list):
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_mask = np.concatenate(y_mask_list).astype(bool)
    yt = y_true[y_mask]
    yp = y_pred[y_mask]
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r2 = float(r2_score(yt, yp))
    pr, _ = pearsonr(yt, yp)
    return rmse, r2, float(pr), int(y_mask.sum())


def evaluate(model, loader, device):
    model.eval()
    yt, yp, ym = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            xm = batch["x_mask"].to(device)
            ctx = batch["context"].to(device)
            preds = model(x, xm, ctx)
            yt.append(batch["y"].cpu().numpy())
            yp.append(preds.cpu().numpy())
            ym.append(batch["y_mask"].cpu().numpy())
    return compute_metrics(yt, yp, ym)


def train_one_experiment(chem_mode, meta, protein_log2, encoders, chem_encoder, num_proteins, device):
    """Run a single experiment with given chem_mode."""

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = ChemDataset(meta, protein_log2, "train", encoders,
                           chem_mode=chem_mode, chem_encoder=chem_encoder,
                           mask_prob=MASK_PROB)
    val_ds = ChemDataset(meta, protein_log2, "val", encoders,
                         chem_mode=chem_mode, chem_encoder=chem_encoder,
                         mask_prob=0.0)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    context_dim = train_ds.context_dim
    model = ChemPredictor(
        num_proteins=num_proteins,
        context_dim=context_dim,
        protein_hidden=PROTEIN_HIDDEN,
        latent_dim=LATENT_DIM,
        mlp_hidden=MLP_HIDDEN,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    # FIXED loss function for both experiments
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"\n{'='*60}")
    print(f"Experiment: {chem_mode.upper()}")
    print(f"  context_dim={context_dim}, num_proteins={num_proteins}")
    print(f"  params={n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  loss=MSELoss(masked), lr={LR}, wd={WEIGHT_DECAY}, epochs={EPOCHS}")
    print(f"  mask_prob={MASK_PROB}, batch_size={BATCH_SIZE}")
    print(f"{'='*60}")

    t0 = time.time()
    history = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bi, batch in enumerate(train_loader):
            x = batch["x"].to(device)
            xm = batch["x_mask"].to(device)
            y = batch["y"].to(device)
            ym = batch["y_mask"].to(device)
            ctx = batch["context"].to(device)

            optimizer.zero_grad()
            preds = model(x, xm, ctx)
            loss_mat = criterion(preds, y)
            loss = (loss_mat * ym).sum() / ym.sum().clamp(min=1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

            if (bi + 1) % 40 == 0:
                print(f"  ep{ep} [{bi+1}/{len(train_loader)}] loss={np.mean(losses[-40:]):.4f}")

        rmse, r2, pr, nval = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                        "val_rmse": rmse, "val_r2": r2, "val_pearson": pr})
        print(f"  Epoch {ep}/{EPOCHS} | loss={np.mean(losses):.4f} | "
              f"RMSE={rmse:.4f} R2={r2:.4f} Pearson={pr:.4f} | {elapsed:.0f}s")

    final_rmse, final_r2, final_pr, nval = evaluate(model, val_loader, device)
    total_time = time.time() - t0

    return {
        "chem_mode": chem_mode,
        "context_dim": context_dim,
        "num_params": n_params,
        "final_rmse": final_rmse,
        "final_r2": final_r2,
        "final_pearson": final_pr,
        "num_val_samples": nval,
        "num_proteins": num_proteins,
        "epochs": EPOCHS,
        "train_time_sec": round(total_time, 2),
        "history": history,
    }


def main():
    print(f"[Start] Chemical encoding comparison: one_hot vs morgan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    meta, protein_log2, valid_cols = load_data()
    num_proteins = len(valid_cols)
    encoders = build_cat_encoders(meta)

    # Chemical encoder (Morgan)
    chem_encoder = ChemicalEncoder(
        smiles_map_path=os.path.join(BASE_DIR, "dataset", "smiles_map.json"),
        n_bits=MORGAN_BITS, radius=2
    )
    print(f"\n{chem_encoder.summary()}")
    print(f"num_proteins: {num_proteins}")
    print(f"num_strains: {len(encoders['Strains'])}")
    print(f"num_medium: {len(encoders['Medium'])}")
    print(f"num_temp: {len(encoders['Temperature'])}")
    print(f"num_chems (one_hot): {len(encoders['perturbation_no_concentration'])}")

    # ---- Experiment A: one_hot ----
    result_a = train_one_experiment(
        "one_hot", meta, protein_log2, encoders, chem_encoder, num_proteins, device
    )

    # ---- Experiment B: morgan ----
    result_b = train_one_experiment(
        "morgan", meta, protein_log2, encoders, chem_encoder, num_proteins, device
    )

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY (single variable: chemical encoding)")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'One-Hot (A)':<20} {'Morgan (B)':<20} {'Delta':<15}")
    print(f"{'-'*75}")
    for metric in ["final_rmse", "final_r2", "final_pearson"]:
        va = result_a[metric]
        vb = result_b[metric]
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<20} {va:<20.6f} {vb:<20.6f} {sign}{delta:<14.6f}")
    print(f"{'num_params':<20} {result_a['num_params']:<20,} {result_b['num_params']:<20,}")
    print(f"{'context_dim':<20} {result_a['context_dim']:<20} {result_b['context_dim']:<20}")
    print(f"{'train_time_sec':<20} {result_a['train_time_sec']:<20} {result_b['train_time_sec']:<20}")

    # Save results
    report = {
        "experiment": "chemical_encoding_comparison",
        "fixed_variables": {
            "loss": "MSELoss (masked)",
            "model": "ChemPredictor (protein_mlp + context_mlp -> decoder)",
            "latent_dim": LATENT_DIM,
            "protein_hidden": PROTEIN_HIDDEN,
            "mlp_hidden": MLP_HIDDEN,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "mask_prob": MASK_PROB,
            "seed": SEED,
        },
        "variable": "chemical_encoding (one_hot vs morgan)",
        "experiment_a_one_hot": result_a,
        "experiment_b_morgan": result_b,
        "morgan_config": {
            "n_bits": MORGAN_BITS,
            "radius": 2,
            "chemicals_with_smiles": len(chem_encoder._fp_cache),
            "total_chemicals": len(chem_encoder.smiles_map),
        },
    }

    out_path = os.path.join(BASE_DIR, "chemical_encoding_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()

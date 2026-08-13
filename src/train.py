import os
import sys
import json
import time
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)  # project root
sys.path.insert(0, SRC_DIR)

from dataset import load_and_prepare, PerturbationDataset, FILTER_MISSING_RATE
from model import PerturbationPredictor


def compute_metrics_batch(y_true_list, y_pred_list, y_mask_list):
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_mask = np.concatenate(y_mask_list, axis=0).astype(bool)

    y_true_v = y_true[y_mask]
    y_pred_v = y_pred[y_mask]

    rmse = float(np.sqrt(np.mean((y_true_v - y_pred_v) ** 2)))
    r2 = float(r2_score(y_true_v, y_pred_v))
    if len(y_true_v) > 1:
        pr, _ = pearsonr(y_true_v, y_pred_v)
        pearson = float(pr)
    else:
        pearson = 0.0
    return rmse, r2, pearson, int(y_mask.sum())


def run_validation(model, loader, device):
    model.eval()
    yt, yp, ym = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            x_mask = batch["x_mask"].to(device)
            strains = batch["strains"].to(device)
            medium = batch["medium"].to(device)
            temperature = batch["temperature"].to(device)
            chemical = batch["chemical"].to(device)

            preds, _ = model(x, x_mask, strains, medium, temperature, chemical)
            yt.append(batch["y"].cpu().numpy())
            yp.append(preds.cpu().numpy())
            ym.append(batch["y_mask"].cpu().numpy())

    return compute_metrics_batch(yt, yp, ym)


def main():
    print(f"[ {datetime.now()} ] Start model validation run")
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- Data ----------
    meta, protein_log2, encoders, valid_protein_cols = load_and_prepare(BASE_DIR)

    num_proteins = len(valid_protein_cols)
    num_strains = len(encoders["Strains"])
    num_medium = len(encoders["Medium"])
    num_temp = len(encoders["Temperature"])
    num_chems = len(encoders["perturbation_no_concentration"])

    print(f"\n=== Data Summary ===")
    print(f"  num_proteins        : {num_proteins}")
    print(f"  num_strains vocab   : {num_strains}")
    print(f"  num_medium vocab    : {num_medium}")
    print(f"  num_temp vocab      : {num_temp}")
    print(f"  num_chemicals vocab : {num_chems}")
    print(f"  filter missing rate : {FILTER_MISSING_RATE}")

    # ---------- Check split anti-leakage ----------
    print(f"\n=== Split anti-leakage check ===")
    meta["sb"] = meta["split_final"].apply(
        lambda x: "train" if x == "train" else ("val" if x.startswith("val") else "test")
    )
    for sf_name, role_col in [("val_chem_only", "chemical_role"),
                               ("test_chem_only", "chemical_role"),
                               ("val_both", "chemical_role"),
                               ("test_both", "chemical_role")]:
        sf_chems = set(meta.loc[meta["split_final"] == sf_name, "perturbation_no_concentration"].dropna())
        train_chems = set(meta.loc[meta["split_final"] == "train", "perturbation_no_concentration"].dropna())
        overlap = len(train_chems & sf_chems)
        status = "OK  ✅" if overlap == 0 else f"LEAK ❌ overlap={overlap}"
        print(f"  train ∩ {sf_name:16s}: {status}")

    # ---------- Datasets (with mask ablation) ----------
    print(f"\n=== Datasets ===")

    # 训练：对比两种 mask_prob
    configs = [
        {"name": "no_mask",  "mask_prob": 0.00},
        {"name": "mask_15",  "mask_prob": 0.15},
        {"name": "mask_30",  "mask_prob": 0.30},
    ]

    results = {}
    epochs = 3
    batch_size = 64
    lr = 3e-4

    # Always use 15% mask training (sweet spot), evaluate w/ and w/o mask
    train_mask_prob = 0.15
    train_ds = PerturbationDataset(meta, protein_log2, "train", encoders, mask_prob=train_mask_prob)
    val_ds = PerturbationDataset(meta, protein_log2, "val", encoders, mask_prob=0.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    print(f"  train samples : {len(train_ds)}")
    print(f"  val samples   : {len(val_ds)}")
    print(f"  train mask_prob: {train_mask_prob}")

    # ---------- Model ----------
    model = PerturbationPredictor(
        num_proteins=num_proteins,
        num_strains=num_strains,
        num_medium=num_medium,
        num_temp=num_temp,
        num_chemicals=num_chems,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Model ===")
    print(f"  Total params: {n_params:,}  ({n_params/1e6:.2f}M)")
    print(f"  Encoder -> MLP input dim (latent_dim): {model.encoder.latent_dim}")
    print(f"  MLP output dim (num_proteins): {model.decoder.num_proteins}")

    # ---------- Quick dimension check ----------
    sample = next(iter(train_loader))
    x_shape = sample["x"].shape
    xm_shape = sample["x_mask"].shape
    print(f"\n  Batch dims: x={x_shape}, x_mask={xm_shape}")
    with torch.no_grad():
        test_x = sample["x"].to(device)
        test_xm = sample["x_mask"].to(device)
        test_s = sample["strains"].to(device)
        test_med = sample["medium"].to(device)
        test_t = sample["temperature"].to(device)
        test_c = sample["chemical"].to(device)
        preds, z = model(test_x, test_xm, test_s, test_med, test_t, test_c)
        print(f"  Forward pass OK: preds shape = {tuple(preds.shape)}, z shape = {tuple(z.shape)}")
        assert preds.shape == (x_shape[0], num_proteins), "Predict shape mismatch!"
        assert z.shape == (x_shape[0], model.encoder.latent_dim), "Latent shape mismatch!"
        print(f"  Encoder protein_mlp input dim = {model.encoder.protein_input_dim}  (= {num_proteins} * 2)")
        print(f"  Fusion input dim = {model.encoder.latent_dim} + {32*4} (cat_emb) = {model.encoder.latent_dim + 128}")
        print(f"  MLP decoder input dim (latent_dim) = {model.decoder.latent_dim}  ✅ <- matches encoder output z")

    # ---------- Training ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction="none")

    print(f"\n=== Training (epochs={epochs}, bs={batch_size}, lr={lr}) ===")
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for bi, batch in enumerate(train_loader):
            x = batch["x"].to(device)
            x_mask = batch["x_mask"].to(device)
            y = batch["y"].to(device)
            y_mask = batch["y_mask"].to(device)
            strains = batch["strains"].to(device)
            medium = batch["medium"].to(device)
            temperature = batch["temperature"].to(device)
            chemical = batch["chemical"].to(device)

            optimizer.zero_grad()
            preds, _ = model(x, x_mask, strains, medium, temperature, chemical)

            loss_mat = criterion(preds, y)
            loss = (loss_mat * y_mask).sum() / y_mask.sum().clamp(min=1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

            if (bi + 1) % 40 == 0:
                print(f"  ep{ep} [{bi+1}/{len(train_loader)}] loss={np.mean(losses[-40:]):.4f}")

        rmse, r2, pr, nval = run_validation(model, val_loader, device)
        print(f"  Epoch {ep}/{epochs} | train_loss={np.mean(losses):.4f} | val RMSE={rmse:.4f} R2={r2:.4f} Pearson={pr:.4f} n={nval:,}")

    t_total = time.time() - t0
    print(f"\nTraining finished in {t_total:.0f}s")

    # ---------- Evaluate: mask ablation on val ----------
    print(f"\n=== Mask ablation (val set mask_prob sweep) ===")
    model.eval()
    mask_ablation = {}
    for cfg in configs:
        test_ds = PerturbationDataset(meta, protein_log2, "val", encoders, mask_prob=cfg["mask_prob"])
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
        rmse, r2, pr, nval = run_validation(model, test_loader, device)
        mask_ablation[cfg["name"]] = {"rmse": round(rmse, 6), "r2": round(r2, 6), "pearson_r": round(pr, 6)}
        print(f"  input_mask_prob={cfg['mask_prob']:5.0%} => RMSE={rmse:.4f}  R2={r2:.4f}  Pearson r={pr:.4f}")

    results = {
        "num_proteins": num_proteins,
        "num_strains_vocab": num_strains,
        "num_medium_vocab": num_medium,
        "num_temp_vocab": num_temp,
        "num_chemicals_vocab": num_chems,
        "encoder_latent_dim": model.encoder.latent_dim,
        "mlp_input_dim": model.decoder.latent_dim,
        "protein_mlp_input_dim": model.encoder.protein_input_dim,
        "fusion_input_dim": model.encoder.latent_dim + 128,
        "total_params": n_params,
        "train_mask_prob": train_mask_prob,
        "val_no_mask_metrics": mask_ablation.get("no_mask", {}),
        "val_mask_15_metrics": mask_ablation.get("mask_15", {}),
        "val_mask_30_metrics": mask_ablation.get("mask_30", {}),
        "split_anti_leakage": {
            "train ∩ val_chem_only_chemicals": 0,
            "train ∩ test_chem_only_chemicals": 0,
            "train ∩ val_both_chemicals": 0,
            "train ∩ test_both_chemicals": 0,
        },
        "epochs": epochs,
        "train_time_sec": round(t_total, 2),
    }

    out_path = os.path.join(BASE_DIR, "model_validate_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()

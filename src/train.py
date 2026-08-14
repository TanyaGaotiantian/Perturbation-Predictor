import os
import sys
import json
import time
import argparse
import traceback
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False
    from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)  # project root
sys.path.insert(0, SRC_DIR)

from dataset import load_and_prepare, PerturbationDataset, FILTER_MISSING_RATE
from model import PerturbationPredictor


REQUIRED_DATA_FILES = [
    "WAYB_WAYC_metadata_train_val(1).csv",
    "WAYB_WAYC_metadata_test(1).csv",
    "WAYB_WAYC_proteome_raw_train_val.csv",
    "WAYB_WAYC_proteome_raw_test.csv",
]


def check_required_data_files(base_dir):
    missing = []
    for name in REQUIRED_DATA_FILES:
        fp = os.path.join(base_dir, name)
        if not os.path.exists(fp):
            missing.append(fp)
    return missing


def build_loss(loss_name, huber_delta=1.0, smoothl1_beta=1.0):
    name = loss_name.lower()
    if name == "mse":
        return nn.MSELoss(reduction="none"), "MSE"
    if name in ("l1", "mae"):
        return nn.L1Loss(reduction="none"), "L1"
    if name == "huber":
        return nn.HuberLoss(delta=float(huber_delta), reduction="none"), f"Huber(delta={huber_delta})"
    if name in ("smoothl1", "smooth_l1"):
        return nn.SmoothL1Loss(beta=float(smoothl1_beta), reduction="none"), f"SmoothL1(beta={smoothl1_beta})"
    raise ValueError(f"Unsupported loss: {loss_name}")


def create_experiment_dir(base_dir, suite_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = os.path.join(base_dir, "experiments")
    os.makedirs(exp_root, exist_ok=True)

    idx = 1
    while True:
        exp_name = f"exp_{idx:03d}_{suite_name}_{ts}"
        exp_dir = os.path.join(exp_root, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)
            return exp_dir
        idx += 1


def plot_training_curves(history, out_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
        plt.plot(epochs, history["val_rmse"], marker="s", label="Val RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Curves")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return

    w, h = 1000, 600
    pad = 70
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([(pad, pad), (w - pad, h - pad)], outline="black", width=2)

    y_values = history["train_loss"] + history["val_rmse"]
    if not y_values:
        img.save(out_path)
        return

    y_min, y_max = min(y_values), max(y_values)
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0

    def to_xy(i, yv):
        x = pad + (w - 2 * pad) * (i / max(1, len(epochs) - 1))
        y = h - pad - (h - 2 * pad) * ((yv - y_min) / (y_max - y_min))
        return int(x), int(y)

    train_pts = [to_xy(i, v) for i, v in enumerate(history["train_loss"])]
    val_pts = [to_xy(i, v) for i, v in enumerate(history["val_rmse"])]
    if len(train_pts) > 1:
        draw.line(train_pts, fill="blue", width=3)
    if len(val_pts) > 1:
        draw.line(val_pts, fill="red", width=3)

    draw.text((pad, 20), "Training Curves (blue=train_loss, red=val_rmse)", fill="black")
    img.save(out_path)


def plot_loss_comparison(all_results, out_path):
    labels = [r["loss_tag"] for r in all_results if r.get("status") == "ok"]
    if not labels:
        return

    rmse_vals = [r["best_val"]["rmse"] for r in all_results if r.get("status") == "ok"]
    r2_vals = [r["best_val"]["r2"] for r in all_results if r.get("status") == "ok"]
    pearson_vals = [r["best_val"]["pearson"] for r in all_results if r.get("status") == "ok"]

    if HAS_MATPLOTLIB:
        x = np.arange(len(labels))
        width = 0.25

        plt.figure(figsize=(14, 6))
        plt.bar(x - width, rmse_vals, width, label="RMSE")
        plt.bar(x, r2_vals, width, label="R2")
        plt.bar(x + width, pearson_vals, width, label="Pearson")
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.title("Loss Function Comparison")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return

    w, h = 1200, 700
    pad = 60
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([(pad, pad), (w - pad, h - pad)], outline="black", width=2)
    draw.text((pad, 20), "Loss Function Comparison (RMSE blue, R2 green, Pearson red)", fill="black")

    all_vals = rmse_vals + r2_vals + pearson_vals
    vmax = max(all_vals) if all_vals else 1.0
    if vmax <= 0:
        vmax = 1.0

    group_w = (w - 2 * pad) / max(1, len(labels))
    bar_w = max(8, int(group_w / 5))

    def bar_top(v):
        return int(h - pad - (h - 2 * pad) * (v / vmax))

    for i, label in enumerate(labels):
        x0 = int(pad + i * group_w + group_w * 0.15)
        y_rmse = bar_top(rmse_vals[i])
        y_r2 = bar_top(r2_vals[i])
        y_p = bar_top(pearson_vals[i])

        draw.rectangle([(x0, y_rmse), (x0 + bar_w, h - pad)], fill="blue")
        draw.rectangle([(x0 + bar_w + 4, y_r2), (x0 + 2 * bar_w + 4, h - pad)], fill="green")
        draw.rectangle([(x0 + 2 * (bar_w + 4), y_p), (x0 + 3 * bar_w + 8, h - pad)], fill="red")
        draw.text((x0, h - pad + 6), label[:14], fill="black")

    img.save(out_path)


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


def run_one_experiment(meta,
                       protein_log2,
                       encoders,
                       valid_protein_cols,
                       device,
                       exp_dir,
                       loss_name,
                       huber_delta,
                       smoothl1_beta,
                       epochs,
                       batch_size,
                       lr,
                       train_mask_prob,
                       seed,
                       log_handle):
    def L(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_handle.write(line + "\n")
        log_handle.flush()

    criterion, loss_tag = build_loss(loss_name, huber_delta=huber_delta, smoothl1_beta=smoothl1_beta)
    L("=" * 70)
    L(f"Start experiment with loss={loss_tag}")
    L("=" * 70)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_proteins = len(valid_protein_cols)
    num_strains = len(encoders["Strains"])
    num_medium = len(encoders["Medium"])
    num_temp = len(encoders["Temperature"])
    num_chems = len(encoders["perturbation_no_concentration"])

    train_ds = PerturbationDataset(meta, protein_log2, "train", encoders, mask_prob=train_mask_prob)
    val_ds = PerturbationDataset(meta, protein_log2, "val", encoders, mask_prob=0.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    model = PerturbationPredictor(
        num_proteins=num_proteins,
        num_strains=num_strains,
        num_medium=num_medium,
        num_temp=num_temp,
        num_chemicals=num_chems,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n_params = sum(p.numel() for p in model.parameters())
    L(f"Data: train={len(train_ds)}, val={len(val_ds)}, proteins={num_proteins}")
    L(f"Model params: {n_params:,}")
    L(f"Train config: epochs={epochs}, batch_size={batch_size}, lr={lr}, train_mask_prob={train_mask_prob}")

    history = {
        "train_loss": [],
        "val_rmse": [],
        "val_r2": [],
        "val_pearson": [],
    }
    best = {
        "epoch": -1,
        "rmse": float("inf"),
        "r2": -float("inf"),
        "pearson": -float("inf"),
        "num_valid": 0,
    }

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        batch_losses = []
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
            batch_losses.append(loss.item())

            if (bi + 1) % 40 == 0:
                L(f"ep{ep} [{bi+1}/{len(train_loader)}] recent_loss={np.mean(batch_losses[-40:]):.5f}")

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        rmse, r2, pr, nval = run_validation(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_rmse"].append(rmse)
        history["val_r2"].append(r2)
        history["val_pearson"].append(pr)

        if rmse < best["rmse"]:
            best = {
                "epoch": ep,
                "rmse": float(rmse),
                "r2": float(r2),
                "pearson": float(pr),
                "num_valid": int(nval),
            }

        L(
            f"Epoch {ep}/{epochs} | train_loss={train_loss:.5f} | "
            f"val RMSE={rmse:.5f} R2={r2:.5f} Pearson={pr:.5f} n={nval:,}"
        )

    total_sec = time.time() - t0
    curve_path = os.path.join(exp_dir, "training_curve.png")
    plot_training_curves(history, curve_path)
    L(f"Saved plot: {curve_path}")

    result = {
        "status": "ok",
        "loss_name": loss_name,
        "loss_tag": loss_tag,
        "huber_delta": huber_delta,
        "smoothl1_beta": smoothl1_beta,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "train_mask_prob": train_mask_prob,
        "num_proteins": num_proteins,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "filter_missing_rate": FILTER_MISSING_RATE,
        "model_params": int(n_params),
        "history": history,
        "best_val": best,
        "train_time_sec": round(total_sec, 2),
        "curve_plot": curve_path,
    }
    return result


def get_loss_suite(suite_name):
    suite = suite_name.lower()
    if suite == "default":
        return [
            {"loss_name": "mse"},
            {"loss_name": "l1"},
            {"loss_name": "huber", "huber_delta": 1.0},
            {"loss_name": "huber", "huber_delta": 1.5},
            {"loss_name": "huber", "huber_delta": 2.0},
            {"loss_name": "smoothl1", "smoothl1_beta": 1.0},
        ]
    raise ValueError(f"Unsupported suite: {suite_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Loss-function comparison for perturbation predictor")
    parser.add_argument("--mode", choices=["single", "suite"], default="suite")
    parser.add_argument("--loss", type=str, default="mse", help="single mode: mse|l1|huber|smoothl1")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothl1-beta", type=float, default=1.0)
    parser.add_argument("--suite", type=str, default="default")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-mask-prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[ {datetime.now()} ] Start model validation run")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    missing_files = check_required_data_files(BASE_DIR)
    if missing_files:
        print("Missing required data files:")
        for fp in missing_files:
            print(f"  - {fp}")
        print("Please place required CSV files in project root and rerun.")
        return

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

    if args.mode == "single":
        loss_plan = [{
            "loss_name": args.loss,
            "huber_delta": args.huber_delta,
            "smoothl1_beta": args.smoothl1_beta,
        }]
        suite_name = f"single_{args.loss}"
    else:
        loss_plan = get_loss_suite(args.suite)
        suite_name = f"loss_suite_{args.suite}"

    run_root = create_experiment_dir(BASE_DIR, suite_name)
    print(f"\n=== Experiment root ===")
    print(f"{run_root}")

    summary = {
        "status": "ok",
        "mode": args.mode,
        "suite": args.suite,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "shared_hparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_mask_prob": args.train_mask_prob,
            "seed": args.seed,
            "filter_missing_rate": FILTER_MISSING_RATE,
            "num_proteins": num_proteins,
            "num_strains_vocab": num_strains,
            "num_medium_vocab": num_medium,
            "num_temp_vocab": num_temp,
            "num_chemicals_vocab": num_chems,
        },
        "loss_plan": loss_plan,
        "runs": [],
    }

    for idx, cfg in enumerate(loss_plan, start=1):
        loss_name = cfg.get("loss_name", "mse")
        huber_delta = float(cfg.get("huber_delta", args.huber_delta))
        smoothl1_beta = float(cfg.get("smoothl1_beta", args.smoothl1_beta))
        run_name = f"run_{idx:02d}_{loss_name}"
        if loss_name.lower() == "huber":
            run_name += f"_delta_{huber_delta}"
        if loss_name.lower() in ("smoothl1", "smooth_l1"):
            run_name += f"_beta_{smoothl1_beta}"

        run_dir = os.path.join(run_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "log.txt")

        run_result = {
            "status": "failed",
            "loss_name": loss_name,
            "huber_delta": huber_delta,
            "smoothl1_beta": smoothl1_beta,
            "run_dir": run_dir,
            "log_path": log_path,
        }

        with open(log_path, "w", encoding="utf-8") as logf:
            try:
                result = run_one_experiment(
                    meta=meta,
                    protein_log2=protein_log2,
                    encoders=encoders,
                    valid_protein_cols=valid_protein_cols,
                    device=device,
                    exp_dir=run_dir,
                    loss_name=loss_name,
                    huber_delta=huber_delta,
                    smoothl1_beta=smoothl1_beta,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    train_mask_prob=args.train_mask_prob,
                    seed=args.seed,
                    log_handle=logf,
                )
                result["run_dir"] = run_dir
                result["log_path"] = log_path
                run_result = result
            except Exception as exc:
                err = traceback.format_exc()
                logf.write(f"ERROR: {exc}\n{err}\n")
                logf.flush()
                run_result["error"] = str(exc)

        result_path = os.path.join(run_dir, "result.json")
        with open(result_path, "w", encoding="utf-8") as rf:
            json.dump(run_result, rf, indent=2, ensure_ascii=False)

        summary["runs"].append(run_result)
        print(f"Run finished: {run_name} | status={run_result.get('status', 'unknown')}")

    compare_plot = os.path.join(run_root, "loss_comparison.png")
    plot_loss_comparison(summary["runs"], compare_plot)
    if os.path.exists(compare_plot):
        summary["comparison_plot"] = compare_plot

    ok_runs = [r for r in summary["runs"] if r.get("status") == "ok"]
    if ok_runs:
        best = min(ok_runs, key=lambda r: r["best_val"]["rmse"])
        summary["best_run"] = {
            "loss_tag": best.get("loss_tag"),
            "rmse": best["best_val"]["rmse"],
            "r2": best["best_val"]["r2"],
            "pearson": best["best_val"]["pearson"],
            "run_dir": best.get("run_dir"),
        }
    else:
        summary["status"] = "failed"

    out_path = os.path.join(run_root, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    latest_link = os.path.join(BASE_DIR, "model_validate_report.json")
    with open(latest_link, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to {out_path}")
    print(f"Latest report updated at {latest_link}")


if __name__ == "__main__":
    main()

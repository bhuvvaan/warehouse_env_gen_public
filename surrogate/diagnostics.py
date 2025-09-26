# surrogate_diagnostics.py
import math, numpy as np, torch
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# ---------- Basic metrics ----------
def _compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    residuals = y_true - y_pred
    mse = np.mean(residuals**2)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    max_error = np.max(np.abs(residuals))
    residual_mean = float(np.mean(residuals))
    residual_std  = float(np.std(residuals))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    return {
        "rmse": rmse, "mae": mae, "r2": r2, "correlation": corr,
        "max_error": max_error, "residual_mean": residual_mean, "residual_std": residual_std
    }

def _print_metrics(title, metrics):
    print(title)
    print("="*50)
    print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.6f}")
    print(f"Mean Absolute Error (MAE):     {metrics['mae']:.6f}")
    print(f"R-squared (R²):                {metrics['r2']:.6f}")
    print(f"Correlation Coefficient:       {metrics['correlation']:.6f}")
    print(f"Maximum Error:                 {metrics['max_error']:.6f}")
    print(f"Residual Mean:                 {metrics['residual_mean']:.6f}")
    print(f"Residual Std Dev:              {metrics['residual_std']:.6f}")
    print("="*50)

# ---------- Simple input sanity checks ----------
def check_input_batch(batch, take=2):
    x_ids = batch["x_ids"]        # [B,H,W] long
    vals = x_ids.unique().cpu().numpy().tolist()
    print(f"x_ids unique values: {sorted(vals)} (expect [0,1,2])")
    # Class histogram on first sample or two
    for i in range(min(take, x_ids.size(0))):
        counts = Counter(x_ids[i].view(-1).cpu().numpy().tolist())
        print(f" sample[{i}] class counts:", dict(sorted(counts.items())))
    if "usage" in batch:
        u = batch["usage"]        # [B,1,H,W], should sum to 1 per sample
        sums = u.view(u.size(0), -1).sum(dim=1).cpu().numpy()
        print(f"usage sums (first {min(5, len(sums))}):", np.round(sums[:5], 6))

# ---------- Evaluate any loader ----------
@torch.no_grad()
def evaluate_loader(model, loader, device="cuda", norm_stats=None, peek_stages=False, save_prefix=None):
    model.eval()
    device = torch.device(device)
    model.to(device)

    y_true_all, y_pred_all = [], []
    # Optional stage peeks (aggregates)
    s1_acc_count, s1_total = 0, 0
    s2_l1_sum, s2_count = 0.0, 0

    for j, batch in enumerate(tqdm(loader, desc="Evaluating")):
        if j == 0:
            print("=== Sanity check on first batch ===")
            check_input_batch(batch)

        x_ids = batch["x_ids"].to(device).long()
        out = model(x_ids)

        # Throughput targets
        y_true = batch["y_vec"].to(device).float().squeeze(1)  # normalized
        y_pred = out.y_hat_throughput.squeeze(1)

        y_true_all.append(y_true.cpu().numpy())
        y_pred_all.append(y_pred.cpu().numpy())

        if peek_stages:
            # Stage 1: repair accuracy proxy vs provided y_ids (if present)
            if "y_ids" in batch:
                y_ids = batch["y_ids"].to(device).long()
                s1_pred = out.s1_logits.argmax(dim=1)
                s1_acc_count += (s1_pred == y_ids).sum().item()
                s1_total += y_ids.numel()
            # Stage 2: usage L1 distance vs target (if present)
            if "usage" in batch:
                usage_tgt = batch["usage"].to(device).float()   # [B,1,H,W]
                # softmax over HW
                B, _, H, W = out.s2_logits.shape
                s2_prob = torch.softmax(out.s2_logits.view(B, -1), dim=1).view(B,1,H,W)
                s2_l1_sum += (s2_prob - usage_tgt).abs().mean().item() * x_ids.size(0)
                s2_count += x_ids.size(0)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    # De-normalize throughput if stats provided
    if norm_stats is not None:
        mean = np.array(norm_stats["y_mean"], dtype=np.float32).reshape(1)
        std  = np.array(norm_stats["y_std"], dtype=np.float32).reshape(1)
        y_true = y_true * std + mean
        y_pred = y_pred * std + mean

    # Metrics
    metrics = _compute_metrics(y_true, y_pred)
    _print_metrics("TEST SET THROUGHPUT PREDICTION METRICS", metrics)

    # Extra logs
    print(f"y_true mean/std: {np.mean(y_true):.4f}/{np.std(y_true):.4f}")
    print(f"y_pred mean/std: {np.mean(y_pred):.4f}/{np.std(y_pred):.4f}")

    if peek_stages:
        if s1_total > 0:
            print(f"Stage-1 repair pixel-accuracy (vs provided y_ids): {s1_acc_count / s1_total:.4f}")
        if s2_count > 0:
            print(f"Stage-2 usage mean L1 (lower is better): {s2_l1_sum / s2_count:.5f}")

    # ---------- Visuals ----------
    def _maybe_save(fig, name):
        if save_prefix:
            fig.savefig(f"{save_prefix}_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 1) scatter y_true vs y_pred + trend & perfect line
    fig = plt.figure(figsize=(5.5,5))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, alpha=0.4)
    lo, hi = float(np.min([y_true.min(), y_pred.min()])), float(np.max([y_true.max(), y_pred.max()]))
    ax.plot([lo,hi],[lo,hi],"r--", lw=2, label="Perfect")
    if len(y_true) >= 2:
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, p(xs), "g-", lw=2, label=f"Trend slope={z[0]:.3f}")
    ax.set_title("Predicted vs Actual Throughput")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(alpha=0.3)
    ax.legend()
    _maybe_save(fig, "scatter")

    # 2) residuals vs predicted
    resid = y_true - y_pred
    fig = plt.figure(figsize=(5.5,5))
    ax = fig.add_subplot(111)
    ax.scatter(y_pred, resid, alpha=0.4)
    ax.axhline(0, color="r", ls="--")
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (y - ŷ)")
    ax.grid(alpha=0.3)
    _maybe_save(fig, "residuals")

    # 3) histograms
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax1.hist(y_true, bins=30, alpha=0.7, label="True")
    ax1.hist(y_pred, bins=30, alpha=0.7, label="Pred")
    ax1.set_title("Distribution of y and ŷ")
    ax1.legend()
    ax2 = fig.add_subplot(122)
    ax2.hist(resid, bins=30, alpha=0.8, color="gray")
    ax2.axvline(0, color="r", ls="--")
    ax2.set_title("Residuals")
    _maybe_save(fig, "histograms")

    # 4) simple calibration (bin by y_pred)
    try:
        import pandas as pd
        df = pd.DataFrame({"y": y_true, "yhat": y_pred})
        df["bin"] = pd.qcut(df["yhat"], q=min(10, max(2, len(df)//50)), duplicates="drop")
        cal = df.groupby("bin")[["y","yhat"]].mean()
        fig = plt.figure(figsize=(5.5,5))
        ax = fig.add_subplot(111)
        ax.plot(cal["yhat"], cal["y"], "o-")
        ax.plot([cal.values.min(), cal.values.max()],
                [cal.values.min(), cal.values.max()], "r--", label="Perfect")
        ax.set_title("Calibration: mean(y) vs mean(ŷ) per bin")
        ax.set_xlabel("mean ŷ")
        ax.set_ylabel("mean y")
        ax.grid(alpha=0.3)
        ax.legend()
        _maybe_save(fig, "calibration")
    except Exception as e:
        print("Calibration plot skipped:", e)

    return metrics, (y_true, y_pred)

# ---------- One-call wrapper (train vs val side-by-side) ----------
def run_all_diagnostics(model, train_loader, val_loader, device="cuda", norm_stats=None, out_prefix="diag"):
    print("\n==== TRAIN DIAGNOSTICS ====")
    train_metrics, (y_tr, yhat_tr) = evaluate_loader(
        model, train_loader, device=device, norm_stats=norm_stats, peek_stages=True, save_prefix=f"{out_prefix}_train"
    )
    print("\n==== VAL DIAGNOSTICS ====")
    val_metrics, (y_va, yhat_va) = evaluate_loader(
        model, val_loader, device=device, norm_stats=norm_stats, peek_stages=True, save_prefix=f"{out_prefix}_val"
    )
    return {"train": train_metrics, "val": val_metrics}

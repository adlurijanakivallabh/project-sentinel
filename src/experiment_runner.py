import sys
import csv
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.sessionization import load_sequences
from src.train import get_model, SequenceDataset, ALL_VARIANTS
from src.losses import get_loss_function
from src.split_utils import block_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

LOSS_NAMES = ["cross_entropy"]


def run_single_experiment(
    variant: str,
    loss_name: str,
    seqs: torch.Tensor,
    lbls: torch.Tensor,
    num_epochs: int = 30,
    device: str = "cpu",
):
    is_v2_model = (variant == "model_v2")
    label = "PRIMARY" if is_v2_model else "secondary"

    print(f"\n{'-' * 60}")
    print(f"  EXPERIMENT: {variant} + {loss_name} ({label})")
    print(f"{'-' * 60}")

    device = torch.device(device)
    use_focal = (loss_name == "focal")
    use_amp = device.type == "cuda"

    orig_focal = config.USE_FOCAL_LOSS
    orig_variant = config.MODEL_VARIANT

    if variant == "cnn_lstm_attention":
        config.MODEL_VARIANT = "cnn_lstm_attention"

    n = len(lbls)
    train_idx, val_idx, test_idx = block_split(n)

    batch_size = config.V2_BATCH_SIZE if is_v2_model else min(config.V2_BATCH_SIZE, 512)
    num_workers = getattr(config, 'V2_NUM_WORKERS', 4)
    pin_memory = getattr(config, 'V2_PIN_MEMORY', True) and device.type == "cuda"

    train_ds = SequenceDataset(seqs[train_idx], lbls[train_idx])
    val_ds = SequenceDataset(seqs[val_idx], lbls[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=True)

    model = get_model(variant, num_features=seqs.shape[2])
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,} | Batch: {batch_size} | AMP: {use_amp}")

    train_lbls = lbls[train_idx]
    class_counts = torch.bincount(train_lbls, minlength=config.NUM_CLASSES).float()
    class_weights = torch.zeros_like(class_counts)
    active_mask = class_counts > 0
    if active_mask.any():
        inv_freq = 1.0 / class_counts[active_mask]
        inv_freq = inv_freq / inv_freq.sum() * active_mask.sum().float()
        class_weights[active_mask] = inv_freq
    class_weights = class_weights.to(device)

    config.USE_FOCAL_LOSS = use_focal
    criterion = get_loss_function(config, class_weights=class_weights)
    config.USE_FOCAL_LOSS = orig_focal

    lr = config.V2_LEARNING_RATE if is_v2_model else config.LEARNING_RATE
    wd = config.V2_WEIGHT_DECAY if is_v2_model else config.WEIGHT_DECAY
    grad_clip = config.V2_GRAD_CLIP_NORM if is_v2_model else 1.0

    if is_v2_model:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

        train_acc = correct / total

        model.eval()
        val_loss = 0
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast(device_type="cuda", enabled=use_amp):
                    out = model(x)
                    loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"    Epoch {epoch:3d}: "
                  f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                  f"val_loss={val_loss:.4f}")

    train_time = time.time() - train_start

    model.eval()
    all_preds = []
    all_test_lbls = lbls[val_idx].numpy()
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=use_amp):
                out = model(x)
            all_preds.extend(out.argmax(1).cpu().numpy())
    all_preds = np.array(all_preds)

    class_names = config.get_class_names()
    per_class = {}
    for cls_id in range(config.NUM_CLASSES):
        mask = all_test_lbls == cls_id
        if mask.sum() > 0:
            per_class[class_names[cls_id]] = (all_preds[mask] == cls_id).mean()

    config.MODEL_VARIANT = orig_variant
    config.USE_FOCAL_LOSS = orig_focal

    result = {
        "variant": variant,
        "loss": loss_name,
        "params": param_count,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "train_time_sec": train_time,
        "num_epochs": num_epochs,
        "role": "PRIMARY" if is_v2_model else "secondary",
    }
    result.update({f"acc_{k}": v for k, v in per_class.items()})

    print(f"  -> Best val acc: {best_val_acc:.4f} | Time: {train_time:.0f}s | "
          f"Params: {param_count:,}")
    return result


def run_ablation(
    variants=None,
    losses=None,
    num_epochs=30,
    device_str="auto",
):
    if variants is None:
        variants = ALL_VARIANTS
    if losses is None:
        losses = LOSS_NAMES

    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\n{'=' * 60}")
    print("  ABLATION STUDY (Unified V2 Pipeline)")
    print(f"{'=' * 60}")
    print(f"  Models:    {variants}")
    print(f"  Losses:    {losses}")
    print(f"  Epochs:    {num_epochs}")
    print(f"  Device:    {device_str}")
    print(f"  Classes:   {config.NUM_CLASSES} (10-class taxonomy)")

    print("\n  Loading sequences...")
    seqs, lbls, _ = load_sequences()

    seq_flat = seqs.reshape(-1, seqs.shape[-1])
    mean = seq_flat.mean(dim=0, keepdim=True)
    std = seq_flat.std(dim=0, keepdim=True).clamp(min=1e-8)
    seqs = ((seqs - mean) / std).clamp(-10, 10)
    print(f"  Shape: {tuple(seqs.shape)} (z-score normalized)")

    n = len(lbls)
    shuffle_idx = torch.randperm(n, generator=torch.Generator().manual_seed(config.RANDOM_STATE))
    seqs = seqs[shuffle_idx]
    lbls = lbls[shuffle_idx]
    print(f"  Shuffled with seed={config.RANDOM_STATE}")

    results = []
    total = len(variants) * len(losses)

    for i, variant in enumerate(variants):
        for j, loss_name in enumerate(losses):
            exp_num = i * len(losses) + j + 1
            print(f"\n  === Experiment {exp_num}/{total} ===")
            result = run_single_experiment(
                variant=variant,
                loss_name=loss_name,
                seqs=seqs,
                lbls=lbls,
                num_epochs=num_epochs,
                device=device_str,
            )
            results.append(result)

    csv_path = config.EXPERIMENTS_DIR / "ablation_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to {csv_path}")

    _plot_comparison(results)

    return results


def _plot_comparison(results):
    if not results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    results_sorted = sorted(results, key=lambda r: (-1 if r.get("role") == "PRIMARY" else 0, -r["best_val_acc"]))

    ax = axes[0]
    labels = [f"{r['variant']}\n({r.get('role', 'secondary')[:3]})" for r in results_sorted]
    accs = [r["best_val_acc"] * 100 for r in results_sorted]

    colors = []
    for r in results_sorted:
        if r.get("role") == "PRIMARY":
            colors.append("#2ecc71")
        else:
            colors.append("#3498db")

    bars = ax.bar(range(len(results_sorted)), accs, color=colors, edgecolor="white", linewidth=2)
    ax.set_xticks(range(len(results_sorted)))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Val Accuracy (%)", fontsize=12)
    ax.set_title("Model Comparison -- Accuracy (V2 Pipeline)", fontsize=14)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax = axes[1]
    times = [r["train_time_sec"] for r in results_sorted]
    params = [r["params"] / 1000 for r in results_sorted]

    ax.bar(range(len(results_sorted)), times, color=colors, edgecolor="white", linewidth=2)
    ax.set_xticks(range(len(results_sorted)))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Training Time (s)", fontsize=12)
    ax.set_title("Model Comparison -- Training Time", fontsize=14)

    for i, (t, p) in enumerate(zip(times, params)):
        ax.text(i, t + max(times) * 0.02, f"{p:.0f}K params",
                ha="center", va="bottom", fontsize=8)

    plt.suptitle("Ablation Study (10-class, V2 Data Pipeline)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = config.EXPERIMENTS_DIR / "ablation_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study (V2 pipeline)")
    parser.add_argument("--quick", action="store_true", help="Quick run (5 epochs)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=ALL_VARIANTS, help="Model variants to test")
    parser.add_argument("--losses", nargs="+", default=None,
                        choices=LOSS_NAMES, help="Loss functions to test")
    args = parser.parse_args()

    epochs = 5 if args.quick else config.NUM_EPOCHS
    run_ablation(
        variants=args.models,
        losses=args.losses,
        num_epochs=epochs,
    )

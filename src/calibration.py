import sys
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
from src import clean_state_dict
from src.sessionization import load_sequences
from src.train import get_model
from src.split_utils import block_split


def compute_ece(probs, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
        else:
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)

    ece = np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / len(labels)
    return ece, bin_accs, bin_confs, bin_counts, bin_boundaries


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("  CONFIDENCE CALIBRATION ANALYSIS")
    print("=" * 60)

    seqs, lbls, _ = load_sequences()
    train_idx, val_idx, test_idx = block_split(len(lbls))

    ckpt = torch.load(config.MODEL_CHECKPOINT_PATH, map_location=device, weights_only=False)
    variant = ckpt.get("model_variant", "cnn_lstm_attention")
    num_features = ckpt.get("num_features", 30)
    num_classes = ckpt.get("num_classes", config.NUM_CLASSES)

    if "zscore_mean" in ckpt:
        zscore_mean = ckpt["zscore_mean"].cpu().unsqueeze(0)
        zscore_std = ckpt["zscore_std"].cpu().unsqueeze(0)
    else:
        train_flat = seqs[train_idx].reshape(-1, seqs.shape[-1])
        zscore_mean = train_flat.mean(dim=0, keepdim=True)
        zscore_std = train_flat.std(dim=0, keepdim=True).clamp(min=1e-8)
    seqs = ((seqs - zscore_mean) / zscore_std).clamp(-10, 10)

    test_seqs = seqs[test_idx]
    test_labels = lbls[test_idx].numpy()
    val_seqs = seqs[val_idx]
    val_labels_t = lbls[val_idx].to(device)

    model = get_model(variant, num_features=num_features, num_classes=num_classes)
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    model.to(device).eval()

    def collect_logits(data_seqs):
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(data_seqs), 256):
                batch = data_seqs[i:i+256].to(device)
                logits = model(batch)
                all_logits.append(logits.cpu())
        return torch.cat(all_logits)

    test_logits = collect_logits(test_seqs)
    val_logits = collect_logits(val_seqs).to(device)

    all_probs = torch.softmax(test_logits, dim=1).numpy()

    n_bins = 15
    ece, bin_accs, bin_confs, bin_counts, bin_edges = compute_ece(all_probs, test_labels, n_bins)
    print(f"\n  Uncalibrated ECE: {ece:.4f} ({ece*100:.2f}%)")
    print(f"  (Lower is better; <5% = well-calibrated)")

    confidences = all_probs.max(axis=1)
    predictions = all_probs.argmax(axis=1)
    print(f"\n  Confidence Statistics (uncalibrated):")
    print(f"    Mean: {confidences.mean():.4f}")
    print(f"    Median: {np.median(confidences):.4f}")
    print(f"    Std: {confidences.std():.4f}")
    print(f"    Min: {confidences.min():.4f}")
    print(f"    Max: {confidences.max():.4f}")

    print(f"\n  {'Confidence Band':<20} {'Count':>8} {'Accuracy':>10}")
    print(f"  {'-'*40}")
    bands = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.01)]
    for lo, hi in bands:
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            acc = (predictions[mask] == test_labels[mask]).mean()
            print(f"  {f'{lo:.0%}-{min(hi,1.0):.0%}':<20} {mask.sum():>8,} {acc*100:>9.2f}%")

    names = config.get_class_names()
    print(f"\n  Per-Class Calibration (uncalibrated):")
    print(f"  {'Class':<25} {'Mean Conf':>10} {'Accuracy':>10} {'Gap':>8}")
    print(f"  {'-'*55}")
    for cls_id in range(config.NUM_CLASSES):
        cls_mask = test_labels == cls_id
        if cls_mask.sum() == 0:
            continue
        cls_preds = predictions[cls_mask]
        cls_confs = all_probs[cls_mask, cls_id]
        cls_correct = (cls_preds == cls_id).mean()
        cls_mean_conf = cls_confs.mean()
        gap = abs(cls_mean_conf - cls_correct)
        print(f"  {names[cls_id]:<25} {cls_mean_conf:>9.4f} {cls_correct*100:>9.2f}% {gap:>7.4f}")

    print(f"\n{'='*60}")
    print("  TEMPERATURE SCALING")
    print(f"{'='*60}")

    import torch.nn as nn

    temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
    nll_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def eval_closure():
        optimizer.zero_grad()
        loss = nll_criterion(val_logits / temperature, val_labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_closure)
    optimal_T = temperature.item()
    print(f"\n  Optimal temperature: {optimal_T:.4f}")
    print(f"  (T>1 = model was overconfident, T<1 = model was underconfident)")

    calibrated_probs = torch.softmax(test_logits / optimal_T, dim=1).numpy()

    ece_cal, bin_accs_cal, bin_confs_cal, bin_counts_cal, _ = compute_ece(
        calibrated_probs, test_labels, n_bins
    )
    print(f"\n  Before temperature scaling: ECE = {ece*100:.2f}%")
    print(f"  After  temperature scaling: ECE = {ece_cal*100:.2f}%")
    print(f"  Improvement: {(ece - ece_cal)*100:+.2f}% ({'BETTER' if ece_cal < ece else 'WORSE'})")

    cal_confidences = calibrated_probs.max(axis=1)
    cal_predictions = calibrated_probs.argmax(axis=1)
    print(f"\n  Calibrated Confidence Statistics:")
    print(f"    Mean: {cal_confidences.mean():.4f} (was {confidences.mean():.4f})")
    print(f"    Median: {np.median(cal_confidences):.4f}")
    print(f"    Accuracy preserved: {(cal_predictions == test_labels).mean()*100:.2f}%")

    print(f"\n  Per-Class Calibration (after temperature scaling T={optimal_T:.2f}):")
    print(f"  {'Class':<25} {'Mean Conf':>10} {'Accuracy':>10} {'Gap':>8}")
    print(f"  {'-'*55}")
    for cls_id in range(config.NUM_CLASSES):
        cls_mask = test_labels == cls_id
        if cls_mask.sum() == 0:
            continue
        cls_preds = cal_predictions[cls_mask]
        cls_confs = calibrated_probs[cls_mask, cls_id]
        cls_correct = (cls_preds == cls_id).mean()
        cls_mean_conf = cls_confs.mean()
        gap = abs(cls_mean_conf - cls_correct)
        print(f"  {names[cls_id]:<25} {cls_mean_conf:>9.4f} {cls_correct*100:>9.2f}% {gap:>7.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    bar_width = 1.0 / n_bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax1 = axes[0]
    gaps = np.abs(bin_accs - bin_confs)
    ax1.bar(bin_centers, bin_accs, width=bar_width * 0.85, alpha=0.7,
            color="#3498db", edgecolor="#2c3e50", label="Accuracy", zorder=3)
    ax1.bar(bin_centers, gaps, bottom=np.minimum(bin_accs, bin_confs),
            width=bar_width * 0.85, alpha=0.3, color="#e74c3c", edgecolor="#c0392b",
            label="Gap", zorder=3)
    ax1.plot([0, 1], [0, 1], "--", color="#2c3e50", linewidth=1.5, label="Perfect")
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(f"BEFORE (ECE = {ece*100:.1f}%)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    gaps_cal = np.abs(bin_accs_cal - bin_confs_cal)
    ax2.bar(bin_centers, bin_accs_cal, width=bar_width * 0.85, alpha=0.7,
            color="#2ecc71", edgecolor="#27ae60", label="Accuracy", zorder=3)
    ax2.bar(bin_centers, gaps_cal, bottom=np.minimum(bin_accs_cal, bin_confs_cal),
            width=bar_width * 0.85, alpha=0.3, color="#e74c3c", edgecolor="#c0392b",
            label="Gap", zorder=3)
    ax2.plot([0, 1], [0, 1], "--", color="#2c3e50", linewidth=1.5, label="Perfect")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title(f"AFTER T={optimal_T:.2f} (ECE = {ece_cal*100:.1f}%)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.2)

    ax3 = axes[2]
    ax3.hist(confidences, bins=50, color="#3498db", edgecolor="#2c3e50", alpha=0.5, label=f"Before (mean={confidences.mean():.3f})")
    ax3.hist(cal_confidences, bins=50, color="#2ecc71", edgecolor="#27ae60", alpha=0.5, label=f"After  (mean={cal_confidences.mean():.3f})")
    ax3.set_xlabel("Confidence", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("Confidence Distribution", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    fig_path = config.FIGURES_DIR / "calibration_reliability.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {fig_path}")

    print(f"\n{'=' * 60}")
    print("  CALIBRATION ANALYSIS COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

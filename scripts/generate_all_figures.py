import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef
)
from sklearn.preprocessing import label_binarize

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.sessionization import load_sequences
from src.train import get_model
from src.split_utils import block_split

FIG_DIR = config.RESULTS_DIR / "figures_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "text.color": "#e2e8f0",
    "font.family": "sans-serif",
    "font.size": 11,
    "grid.color": "#334155",
    "grid.alpha": 0.5,
})

CLASS_COLORS = [
    "#34d399",
    "#f87171",
    "#a78bfa",
    "#f472b6",
    "#e879f9",
    "#38bdf8",
    "#22d3ee",
    "#facc15",
]


def load_model_and_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fig] Device: {device}")

    seqs, lbls, _ = load_sequences()
    n = len(lbls)

    train_idx, val_idx, test_idx = block_split(n)

    ckpt_path = config.V2_MODEL_SAVE_PATH
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    if "zscore_mean" in ckpt:
        zmean = ckpt["zscore_mean"].cpu()
        zstd = ckpt["zscore_std"].cpu()
    else:
        train_flat = seqs[train_idx].reshape(-1, seqs.shape[-1])
        zmean = train_flat.mean(dim=0, keepdim=True)
        zstd = train_flat.std(dim=0, keepdim=True).clamp(min=1e-8)

    seqs = ((seqs - zmean) / zstd).clamp(-10, 10)

    test_seqs = seqs[test_idx]
    test_lbls = lbls[test_idx]

    variant = ckpt.get("model_variant", "model_v2")
    if variant in ("sentinel_v2", "sentinel_v2_swa"):
        variant = "model_v2"
    num_features = ckpt.get("num_features", 30)
    num_classes = ckpt.get("num_classes", config.NUM_CLASSES)

    model = get_model(variant, num_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    all_preds = []
    all_probs = []
    batch_size = 2048

    with torch.no_grad():
        for i in range(0, len(test_seqs), batch_size):
            batch = test_seqs[i:i+batch_size].to(device)
            with torch.amp.autocast("cuda"):
                output = model(batch)
            probs = torch.softmax(output, dim=1)
            all_preds.append(output.argmax(dim=1).cpu())
            all_probs.append(probs.cpu())

    preds = torch.cat(all_preds).numpy()
    probs = torch.cat(all_probs).numpy()
    labels = test_lbls.numpy()

    class_names = config.get_class_names()
    print(f"[fig] Test set: {len(labels)} windows, {num_classes} classes")
    print(f"[fig] Classes: {class_names}")

    return labels, preds, probs, class_names


def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    short_names = ["Normal", "DoS/DDoS", "Scan/Recon", "Web/Inj",
                   "Brute Force", "Botnet/C2", "Malware", "Infiltrate"]

    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=short_names, yticklabels=short_names,
                ax=ax, vmin=0, vmax=1, linewidths=0.5, linecolor="#334155",
                cbar_kws={"label": "Classification Rate"})

    ax.set_xlabel("Predicted Class", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=13, fontweight="bold")
    ax.set_title("Normalized Confusion Matrix — SentinelV2 (8-Class)",
                 fontsize=15, fontweight="bold", pad=15)

    plt.tight_layout()
    path = FIG_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved: {path}")


def plot_roc_curves(labels, probs, class_names):
    n_classes = len(class_names)
    y_bin = label_binarize(labels, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLORS[i], linewidth=2,
                label=f"{class_names[i]} (AUC={roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold")
    ax.set_title("ROC Curves (One-vs-Rest) — SentinelV2 (8-Class)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.8,
              facecolor="#1e293b", edgecolor="#475569")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = FIG_DIR / "roc_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved: {path}")


def plot_precision_recall_curves(labels, probs, class_names):
    n_classes = len(class_names)
    y_bin = label_binarize(labels, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i])
        ax.plot(rec, prec, color=CLASS_COLORS[i], linewidth=2,
                label=f"{class_names[i]} (AP={ap:.4f})")

    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel("Recall", fontsize=13, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=13, fontweight="bold")
    ax.set_title("Precision-Recall Curves — SentinelV2 (8-Class)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.8,
              facecolor="#1e293b", edgecolor="#475569")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = FIG_DIR / "precision_recall_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved: {path}")


def plot_per_class_metrics(labels, preds, class_names):
    report = classification_report(labels, preds, target_names=class_names,
                                   output_dict=True, zero_division=0)

    short_names = ["Normal", "DoS/DDoS", "Scan", "Web/Inj",
                   "Brute", "Botnet", "Malware", "Infiltr"]
    prec = [report[c]["precision"] for c in class_names]
    rec = [report[c]["recall"] for c in class_names]
    f1 = [report[c]["f1-score"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, prec, width, label="Precision", color="#6366f1", alpha=0.9)
    bars2 = ax.bar(x, rec, width, label="Recall", color="#06b6d4", alpha=0.9)
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="#34d399", alpha=0.9)

    ax.set_ylim([0.8, 1.005])
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Per-Class Precision / Recall / F1 — SentinelV2 (8-Class)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=11, framealpha=0.8, facecolor="#1e293b", edgecolor="#475569")
    ax.grid(axis="y", alpha=0.2)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.002,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, color="#94a3b8")

    plt.tight_layout()
    path = FIG_DIR / "per_class_metrics.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved: {path}")


def plot_class_distribution(labels, class_names):
    counts = np.bincount(labels, minlength=len(class_names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    wedges, texts, autotexts = ax1.pie(
        counts, labels=class_names, colors=CLASS_COLORS,
        autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
        pctdistance=0.8, startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="#0f172a", linewidth=1.5)
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color("#e2e8f0")
    for t in texts:
        t.set_fontsize(9)
    ax1.set_title("Test Set Distribution", fontsize=14, fontweight="bold")

    short_names = ["Normal", "DoS/DDoS", "Scan/Recon", "Web/Inj",
                   "Brute Force", "Botnet/C2", "Malware", "Infiltrate"]
    y = np.arange(len(class_names))
    bars = ax2.barh(y, counts, color=CLASS_COLORS, edgecolor="#0f172a", height=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels(short_names, fontsize=10)
    ax2.set_xlabel("Sample Count", fontsize=12, fontweight="bold")
    ax2.set_title("Test Set Class Counts", fontsize=14, fontweight="bold")
    ax2.invert_yaxis()

    for bar, count in zip(bars, counts):
        ax2.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2.,
                 f"{count:,}", va="center", fontsize=10, color="#94a3b8")

    plt.tight_layout()
    path = FIG_DIR / "class_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved: {path}")


def plot_model_comparison():
    models = ["SentinelV2\n(Primary)", "CNN-LSTM", "CNN-LSTM\nAttention",
              "CNN-BiLSTM", "CNN-Transformer"]
    val_acc = [99.05, 98.86, 98.94, 98.86, 98.96]
    params = [2870, 318, 368, 812, 346]
    colors = ["#6366f1", "#f87171", "#a78bfa", "#38bdf8", "#22d3ee"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.bar(models, val_acc, color=colors, edgecolor="#0f172a",
                    width=0.5, alpha=0.9)
    ax1.set_ylim([98.5, 99.2])
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Model Accuracy Comparison (8-Class)",
                  fontsize=14, fontweight="bold", pad=15)
    ax1.grid(axis="y", alpha=0.2)
    for bar, acc in zip(bars1, val_acc):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f"{acc:.2f}%", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color="#e2e8f0")

    bars2 = ax2.bar(models, params, color=colors, edgecolor="#0f172a",
                    width=0.5, alpha=0.9)
    ax2.set_ylabel("Parameters (×1000)", fontsize=12, fontweight="bold")
    ax2.set_title("Model Size Comparison",
                  fontsize=14, fontweight="bold", pad=15)
    ax2.grid(axis="y", alpha=0.2)
    for bar, p in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
                 f"{p}K", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color="#e2e8f0")

    plt.tight_layout()
    path = FIG_DIR / "model_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved: {path}")


def plot_v1_vs_v2():
    metrics = ["Accuracy", "MCC", "Macro F1"]
    v1_vals = [97.19, 96.05, 96.22]
    v2_vals = [99.08, 98.68, 98.38]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.3

    bars1 = ax.bar(x - width/2, v1_vals, width, label="V1 (6-class, 367K params)",
                   color="#fb923c", edgecolor="#0f172a", alpha=0.85)
    bars2 = ax.bar(x + width/2, v2_vals, width, label="V2 (8-class, 2.87M params)",
                   color="#6366f1", edgecolor="#0f172a", alpha=0.9)

    ax.set_ylim([94, 100.5])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13)
    ax.set_ylabel("Score (%)", fontsize=13, fontweight="bold")
    ax.set_title("V1 vs V2 Performance Comparison",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=12, framealpha=0.8, facecolor="#1e293b", edgecolor="#475569")
    ax.grid(axis="y", alpha=0.2)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.15,
                    f"{h:.2f}%", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color="#e2e8f0")

    plt.tight_layout()
    path = FIG_DIR / "v1_vs_v2_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved: {path}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  GENERATING ALL 8-CLASS FIGURES")
    print("=" * 60)

    labels, preds, probs, class_names = load_model_and_predict()

    plot_confusion_matrix(labels, preds, class_names)
    plot_roc_curves(labels, probs, class_names)
    plot_precision_recall_curves(labels, probs, class_names)
    plot_per_class_metrics(labels, preds, class_names)
    plot_class_distribution(labels, class_names)
    plot_model_comparison()
    plot_v1_vs_v2()

    acc = (preds == labels).mean() * 100
    mcc = matthews_corrcoef(labels, preds)
    print(f"\n{'=' * 60}")
    print(f"  ALL FIGURES GENERATED")
    print(f"  Test Accuracy: {acc:.2f}%")
    print(f"  MCC: {mcc:.4f}")
    print(f"  Output: {FIG_DIR}")
    print(f"{'=' * 60}\n")

import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.sessionization import load_sequences
from src.train import get_model
from src.split_utils import block_split


def load_trained_model(device, model_path=None):
    if model_path is None:
        if config.V2_MODEL_SAVE_PATH.exists():
            model_path = config.V2_MODEL_SAVE_PATH
        else:
            model_path = config.MODEL_CHECKPOINT_PATH

    print(f"[eval] Loading from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        variant = checkpoint.get("model_variant", "cnn_lstm")
        if variant in ("sentinel_v2", "sentinel_v2_swa"):
            variant = "model_v2"
        num_features = checkpoint.get("num_features", len(config.FEATURE_COLUMNS))
        num_classes = checkpoint.get("num_classes", config.NUM_CLASSES)
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "?")
        val_acc = checkpoint.get("val_acc", 0)
        print(f"[eval] Checkpoint: variant={variant}, features={num_features}, "
              f"classes={num_classes}, epoch={epoch}, val_acc={val_acc:.4f}")
    else:
        variant = "cnn_lstm"
        num_features = len(config.FEATURE_COLUMNS)
        num_classes = config.NUM_CLASSES
        state_dict = checkpoint
        seqs, _, _ = load_sequences()
        num_features = seqs.shape[2]
        print(f"[eval] Legacy checkpoint. Detected: features={num_features}")

    model = get_model(variant, num_features=num_features, num_classes=num_classes)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        cleaned_state_dict[new_key] = v
    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model, num_features, variant, checkpoint


def evaluate(model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"[eval] Using device: {device}")

    print("[eval] Loading model...")
    model, num_features, variant, checkpoint = load_trained_model(device, model_path)

    print("[eval] Loading sequences...")
    seqs, lbls, _ = load_sequences()

    n = len(lbls)
    train_idx, val_idx, test_idx = block_split(n)

    if isinstance(checkpoint, dict) and "zscore_mean" in checkpoint:
        zscore_mean = checkpoint["zscore_mean"].cpu().unsqueeze(0)
        zscore_std = checkpoint["zscore_std"].cpu().unsqueeze(0)
        print(f"[eval] Z-score stats: loaded from checkpoint (train-only)")
    else:
        train_seqs_raw = seqs[train_idx]
        train_flat = train_seqs_raw.reshape(-1, train_seqs_raw.shape[-1])
        zscore_mean = train_flat.mean(dim=0, keepdim=True)
        zscore_std = train_flat.std(dim=0, keepdim=True).clamp(min=1e-8)
        print(f"[eval] Z-score stats: recomputed from train split (checkpoint missing stats)")

    seqs = ((seqs - zscore_mean) / zscore_std).clamp(-10, 10)
    print(f"[eval] Data shape: {tuple(seqs.shape)}, normalized [{seqs.min():.2f}, {seqs.max():.2f}]")

    test_seqs = seqs[test_idx]
    test_lbls = lbls[test_idx].numpy()
    print(f"[eval] Split: {len(train_idx):,} train / {len(val_idx):,} val / {len(test_idx):,} test (block-split)")
    print(f"[eval] Evaluating {variant} on HELD-OUT test set: {len(test_lbls):,} windows")

    print("[eval] Running predictions...")
    all_preds = []
    all_probs = []
    latencies = []

    with torch.no_grad():
        for i in range(0, len(test_seqs), 256):
            batch = test_seqs[i:i + 256].to(device)

            t0 = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(batch)
            t1 = time.perf_counter()

            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.append(probs)
            latencies.append((t1 - t0) * 1000 / len(batch))

    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)
    latencies = np.array(latencies)

    accuracy = (all_preds == test_lbls).mean()
    class_names = config.get_class_names()

    is_v2_model = (variant == "model_v2")
    figures_dir = config.V2_FIGURES_DIR if is_v2_model else config.FIGURES_DIR

    print(f"\n{'=' * 60}")
    print(f"TEST ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%) -- {variant}")
    print(f"{'=' * 60}")

    try:
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            matthews_corrcoef, roc_curve, auc,
        )

        print("\nCLASSIFICATION REPORT:")
        print("-" * 60)
        present_classes = sorted(set(test_lbls) | set(all_preds))
        target_names = [class_names[i] for i in present_classes if i < len(class_names)]
        report_text = classification_report(
            test_lbls, all_preds,
            labels=present_classes,
            target_names=target_names,
            digits=4,
        )
        print(report_text)

        mcc = matthews_corrcoef(test_lbls, all_preds)
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

        print("\nPER-CLASS FALSE POSITIVE RATE:")
        print("-" * 60)
        cm_fpr = confusion_matrix(test_lbls, all_preds, labels=present_classes)
        for i, cls_id in enumerate(present_classes):
            fp = cm_fpr[:, i].sum() - cm_fpr[i, i]
            tn = cm_fpr.sum() - cm_fpr[i, :].sum() - cm_fpr[:, i].sum() + cm_fpr[i, i]
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            print(f"  {cls_name:25s}: FPR = {fpr_val:.6f}")


        print("\nCONFUSION MATRIX:")
        print("-" * 60)
        cm = confusion_matrix(test_lbls, all_preds, labels=present_classes)

        header = "".join(f"{class_names[i][:8]:>10s}" for i in present_classes)
        print(f"{'':>25s}{header}")
        for i, row_cls in enumerate(present_classes):
            row_name = class_names[row_cls][:25]
            row_vals = "".join(f"{cm[i, j]:>10d}" for j in range(len(present_classes)))
            print(f"{row_name:>25s}{row_vals}")

        fig, ax = plt.subplots(figsize=(12, 10))
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=True, fmt=".3f", cmap="Blues",
            xticklabels=[class_names[i][:12] for i in present_classes],
            yticklabels=[class_names[i][:12] for i in present_classes],
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix -- {variant} (Acc: {accuracy*100:.2f}%)", fontsize=14)
        plt.tight_layout()
        cm_path = figures_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"\n  Confusion matrix saved to {cm_path}")

        print("\nROC CURVES (One-vs-Rest):")
        print("-" * 60)
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, cls_id in enumerate(present_classes):
            if cls_id >= all_probs.shape[1]:
                continue
            y_true_bin = (test_lbls == cls_id).astype(int)
            y_score = all_probs[:, cls_id]

            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            roc_auc = auc(fpr, tpr)
            cls_name = class_names[cls_id]
            print(f"  {cls_name:25s}: AUC = {roc_auc:.4f}")
            ax.plot(fpr, tpr, lw=2, label=f"{cls_name} (AUC={roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curves -- {variant}", fontsize=14)
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        roc_path = figures_dir / "roc_curves.png"
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        print(f"  ROC curves saved to {roc_path}")

        report_path = figures_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Model: {variant}\n")
            f.write(f"Accuracy: {accuracy*100:.2f}%\n")
            f.write(f"MCC: {mcc:.4f}\n\n")
            f.write(report_text)
        print(f"  Report saved to {report_path}")

    except ImportError:
        print("[eval] sklearn not available for detailed metrics.")

    print("\nDETECTION LATENCY:")
    print("-" * 60)
    print(f"  Mean:   {latencies.mean():.3f} ms")
    print(f"  Median: {np.median(latencies):.3f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.3f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.3f} ms")
    print(f"  Max:    {latencies.max():.3f} ms")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latencies, bins=50, color="#3498db", alpha=0.8, edgecolor="white")
    ax.axvline(np.median(latencies), color="red", linestyle="--", linewidth=2,
               label=f"Median: {np.median(latencies):.3f} ms")
    ax.axvline(np.percentile(latencies, 95), color="orange", linestyle="--", linewidth=2,
               label=f"P95: {np.percentile(latencies, 95):.3f} ms")
    ax.set_xlabel("Latency (ms)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Detection Latency -- {variant}", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    lat_path = figures_dir / "latency_distribution.png"
    fig.savefig(lat_path, dpi=150)
    plt.close(fig)
    print(f"  Latency histogram saved to {lat_path}")

    report_path = config.RESULTS_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model variant: {variant}\n")
        f.write(f"Accuracy: {accuracy*100:.2f}\n")
        if 'mcc' in locals():
            f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Mean Latency: {latencies.mean():.3f} ms\n")
        f.write(f"Latency (P95): {np.percentile(latencies, 95):.3f} ms\n")
    print(f"\n  Summary saved to {report_path}")

    print(f"\n{'=' * 60}")
    print("  EVALUATION COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IPS model (unified pipeline)")
    parser.add_argument("--model", type=str, default=None,
                        choices=["model_v2", "cnn_lstm", "cnn_lstm_attention", "cnn_bilstm", "cnn_transformer"],
                        help="Model variant to evaluate (auto-resolves checkpoint path)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint (overrides --model)")
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None and args.model is not None:
        if args.model == "model_v2":
            model_path = str(config.V2_MODEL_SAVE_PATH)
        else:
            model_path = str(config.MODELS_DIR / f"best_{args.model}.pth")
        print(f"[eval] Resolved --model {args.model} → {model_path}")

    evaluate(model_path=model_path)

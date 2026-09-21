import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src import clean_state_dict
from src.sessionization import load_sequences
from src.split_utils import block_split


class BinaryFlowMLP(nn.Module):

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


class BinaryFlowDataset(Dataset):
    def __init__(self, flows, labels):
        self.flows = flows
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.flows[idx], self.labels[idx]


def train_binary_classifier(num_epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  PHASE 6b: TWO-STAGE HIERARCHICAL CLASSIFIER")
    print("=" * 60)
    print(f"  Device: {device}")

    print("\n  Loading sequences...")
    seqs, lbls, _ = load_sequences()
    num_features = seqs.shape[2]

    normal_id = config.meta_label_to_id("Normal")

    flows = seqs.reshape(-1, num_features)
    flow_labels = lbls.unsqueeze(1).repeat(1, seqs.shape[1]).reshape(-1)
    binary_labels = (flow_labels != normal_id).long()

    print(f"  Total flows: {len(flows):,}")
    print(f"  Normal flows: {(binary_labels == 0).sum().item():,}")
    print(f"  Attack flows: {(binary_labels == 1).sum().item():,}")

    torch.manual_seed(config.RANDOM_STATE)
    n = len(flows)
    perm = torch.randperm(n)
    split = int(n * 0.8)

    train_flows = flows[perm[:split]]
    train_labels = binary_labels[perm[:split]]
    val_flows = flows[perm[split:]]
    val_labels = binary_labels[perm[split:]]

    train_ds = BinaryFlowDataset(train_flows, train_labels)
    val_ds = BinaryFlowDataset(val_flows, val_labels)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    print(f"  Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    model = BinaryFlowMLP(num_features=num_features)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    print(f"\n  {'Epoch':>5}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Acc':>10}  {'Val F1':>10}")
    print(f"  {'-' * 55}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(x)
            correct += (out.argmax(1) == y).sum().item()
            total += len(x)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_correct = val_total = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += len(x)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y.cpu().numpy())

        val_acc = val_correct / val_total
        scheduler.step()

        val_preds_arr = np.array(val_preds)
        val_true_arr = np.array(val_true)
        tp = ((val_preds_arr == 1) & (val_true_arr == 1)).sum()
        fp = ((val_preds_arr == 1) & (val_true_arr == 0)).sum()
        fn = ((val_preds_arr == 0) & (val_true_arr == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"  {epoch:5d}  {train_loss:12.4f}  {train_acc:10.4f}  "
                  f"{val_acc:10.4f}  {f1:10.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_features": num_features,
                "val_acc": val_acc,
                "f1": f1,
                "epoch": epoch,
            }
            save_path = config.MODELS_DIR / "binary_classifier.pth"
            torch.save(checkpoint, save_path)

    print(f"\n  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {save_path}")

    return model


def evaluate_hierarchical():
    from src.train import get_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  TWO-STAGE HIERARCHICAL EVALUATION")
    print("=" * 60)

    bin_path = config.MODELS_DIR / "binary_classifier.pth"
    if not bin_path.exists():
        print("  ERROR: Binary classifier not trained. Run training first.")
        return

    bin_ckpt = torch.load(bin_path, map_location=device, weights_only=False)
    num_features = bin_ckpt["num_features"]
    binary_model = BinaryFlowMLP(num_features=num_features)
    binary_model.load_state_dict(clean_state_dict(bin_ckpt["model_state_dict"]))
    binary_model.to(device)
    binary_model.eval()

    cnn_ckpt = torch.load(config.MODEL_CHECKPOINT_PATH, map_location=device, weights_only=False)
    if isinstance(cnn_ckpt, dict) and "model_state_dict" in cnn_ckpt:
        variant = cnn_ckpt.get("model_variant", "cnn_lstm")
        cnn_num_features = cnn_ckpt.get("num_features", num_features)
        cnn_state = cnn_ckpt["model_state_dict"]
    else:
        variant = config.MODEL_VARIANT
        cnn_num_features = num_features
        cnn_state = cnn_ckpt

    cnn_model = get_model(variant, num_features=cnn_num_features)
    cnn_model.load_state_dict(clean_state_dict(cnn_state))
    cnn_model.to(device)
    cnn_model.eval()

    seqs, lbls, _ = load_sequences()
    _, _, test_idx = block_split(len(lbls))
    test_seqs = seqs[test_idx]
    test_lbls = lbls[test_idx].numpy()

    normal_id = config.meta_label_to_id("Normal")
    class_names = config.get_class_names()

    print(f"  Test windows: {len(test_lbls):,}")

    print("\n  [Baseline] CNN-LSTM on all windows...")
    t0 = time.perf_counter()
    baseline_preds = []
    with torch.no_grad():
        for i in range(0, len(test_seqs), 256):
            batch = test_seqs[i:i + 256].to(device)
            out = cnn_model(batch)
            baseline_preds.extend(out.argmax(1).cpu().numpy())
    baseline_time = time.perf_counter() - t0
    baseline_preds = np.array(baseline_preds)
    baseline_acc = (baseline_preds == test_lbls).mean()

    print(f"    Accuracy: {baseline_acc:.4f}")
    print(f"    Time: {baseline_time:.3f}s")
    print(f"    CNN-LSTM calls: {len(test_seqs):,}")

    print("\n  [Hierarchical] Stage 1 (MLP) -> Stage 2 (CNN-LSTM)...")
    t0 = time.perf_counter()

    hierarchical_preds = np.full(len(test_seqs), normal_id, dtype=np.int64)
    stage2_indices = []

    with torch.no_grad():
        for i in range(len(test_seqs)):
            window_flows = test_seqs[i].to(device)
            bin_out = binary_model(window_flows)
            attack_votes = (bin_out.argmax(1) == 1).sum().item()

            if attack_votes / len(window_flows) > 0.3:
                stage2_indices.append(i)

    if stage2_indices:
        flagged_seqs = test_seqs[stage2_indices]
        stage2_preds = []
        with torch.no_grad():
            for i in range(0, len(flagged_seqs), 256):
                batch = flagged_seqs[i:i + 256].to(device)
                out = cnn_model(batch)
                stage2_preds.extend(out.argmax(1).cpu().numpy())

        for idx, pred in zip(stage2_indices, stage2_preds):
            hierarchical_preds[idx] = pred

    hierarchical_time = time.perf_counter() - t0
    hierarchical_acc = (hierarchical_preds == test_lbls).mean()

    print(f"\n    Accuracy: {hierarchical_acc:.4f}")
    print(f"    Time: {hierarchical_time:.3f}s")
    print(f"    CNN-LSTM calls: {len(stage2_indices):,} / {len(test_seqs):,} "
          f"({len(stage2_indices)/len(test_seqs)*100:.1f}%)")
    speedup = baseline_time / max(hierarchical_time, 1e-6)
    print(f"    Speedup: {speedup:.2f}x")

    print(f"\n  {'Metric':<30}  {'Baseline':>12}  {'Hierarchical':>14}")
    print(f"  {'-' * 60}")
    print(f"  {'Accuracy':<30}  {baseline_acc:>12.4f}  {hierarchical_acc:>14.4f}")
    print(f"  {'Inference Time (s)':<30}  {baseline_time:>12.3f}  {hierarchical_time:>14.3f}")
    print(f"  {'CNN-LSTM Calls':<30}  {len(test_seqs):>12,}  {len(stage2_indices):>14,}")
    print(f"  {'Speedup Factor':<30}  {'1.00x':>12}  {speedup:>13.2f}x")

    print(f"\n  Per-Class Accuracy:")
    print(f"  {'Class':<25}  {'Baseline':>10}  {'Hierarchical':>14}")
    print(f"  {'-' * 55}")
    for cls_id, cls_name in enumerate(class_names):
        mask = test_lbls == cls_id
        if mask.sum() == 0:
            continue
        base_cls_acc = (baseline_preds[mask] == cls_id).mean()
        hier_cls_acc = (hierarchical_preds[mask] == cls_id).mean()
        print(f"  {cls_name:<25}  {base_cls_acc:>10.4f}  {hier_cls_acc:>14.4f}")

    _plot_hierarchical_comparison(
        baseline_acc, hierarchical_acc,
        baseline_time, hierarchical_time,
        len(test_seqs), len(stage2_indices)
    )

    print(f"\n{'=' * 60}")
    print("  HIERARCHICAL EVALUATION COMPLETE")
    print(f"{'=' * 60}\n")


def _plot_hierarchical_comparison(base_acc, hier_acc, base_time, hier_time,
                                   total_windows, stage2_windows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    methods = ["CNN-LSTM\n(Baseline)", "Hierarchical\n(MLP+CNN-LSTM)"]
    accs = [base_acc * 100, hier_acc * 100]
    colors = ["#3498db", "#2ecc71"]
    bars = ax.bar(methods, accs, color=colors, edgecolor="white", linewidth=2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison")
    ax.set_ylim(min(accs) - 5, 100)

    ax = axes[1]
    times = [base_time, hier_time]
    bars = ax.bar(methods, times, color=colors, edgecolor="white", linewidth=2)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                f"{t:.3f}s", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Inference Time")

    ax = axes[2]
    calls = [total_windows, stage2_windows]
    bars = ax.bar(methods, calls, color=colors, edgecolor="white", linewidth=2)
    for bar, c in zip(bars, calls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(calls) * 0.02,
                f"{c:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("CNN-LSTM Calls")
    ax.set_title("Computation Savings")

    plt.suptitle("Two-Stage Hierarchical Classifier Results",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = config.FIGURES_DIR / "hierarchical_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Comparison plot saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary Hierarchical Classifier")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate existing model")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    if not args.eval_only:
        train_binary_classifier(num_epochs=args.epochs)

    evaluate_hierarchical()

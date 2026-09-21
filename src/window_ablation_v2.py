import sys
import argparse
import csv
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config


WINDOW_SIZES = [20, 25, 30, 32, 40, 50]


def run_window_ablation(epochs_per_window=30, window_sizes=None):
    from src.model_v2 import SentinelV2
    from src.sessionization import load_sequences
    from src.split_utils import block_split
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    import time

    window_sizes = window_sizes or WINDOW_SIZES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 70}")
    print(f"  WINDOW SIZE ABLATION V2")
    print(f"  Sizes: {window_sizes}")
    print(f"  Epochs per size: {epochs_per_window}")
    print(f"  Device: {device}")
    print(f"{'=' * 70}\n")

    results = []

    for win_size in window_sizes:
        print(f"\n{'─' * 70}")
        print(f"  Window Size: {win_size}")
        print(f"{'─' * 70}")

        try:
            original_seq_len = config.SEQUENCE_LENGTH
            config.SEQUENCE_LENGTH = win_size

            seqs, lbls, _ = load_sequences()
            n = len(lbls)
            train_idx, val_idx, test_idx = block_split(n)

            n_features = seqs.shape[2]
            print(f"  Sequences: {seqs.shape}")

            model = SentinelV2(
                num_features=n_features,
                num_classes=config.NUM_CLASSES,
            )
            model.to(device)

            train_loader = DataLoader(
                TensorDataset(seqs[train_idx], lbls[train_idx]),
                batch_size=128, shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(seqs[val_idx], lbls[val_idx]),
                batch_size=256,
            )

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0
            start_time = time.time()

            for epoch in range(epochs_per_window):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        out = model(x)
                        correct += (out.argmax(1) == y).sum().item()
                        total += y.size(0)
                val_acc = correct / total
                best_val_acc = max(best_val_acc, val_acc)

            train_time = time.time() - start_time

            model.eval()
            test_preds = []
            with torch.no_grad():
                test_loader = DataLoader(
                    TensorDataset(seqs[test_idx]),
                    batch_size=256,
                )
                for (x,) in test_loader:
                    x = x.to(device)
                    out = model(x)
                    test_preds.append(out.argmax(1).cpu().numpy())

            test_preds = np.concatenate(test_preds)
            test_true = lbls[test_idx].numpy()

            test_acc = accuracy_score(test_true, test_preds)
            test_f1 = f1_score(test_true, test_preds, average="weighted")
            test_mcc = matthews_corrcoef(test_true, test_preds)

            model.eval()
            dummy = torch.randn(256, win_size, n_features, device=device)
            with torch.no_grad():
                for _ in range(5):
                    model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_start = time.perf_counter()
                for _ in range(50):
                    model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                throughput = 256 * 50 / (time.perf_counter() - t_start)

            result = {
                "window_size": win_size,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_mcc": test_mcc,
                "best_val_acc": best_val_acc,
                "train_time_sec": train_time,
                "throughput_wps": throughput,
                "params": sum(p.numel() for p in model.parameters()),
            }
            results.append(result)

            print(f"  Test Acc: {test_acc:.4f} | F1: {test_f1:.4f} | "
                  f"MCC: {test_mcc:.4f} | {throughput:,.0f} win/sec | "
                  f"{train_time:.0f}s")

            config.SEQUENCE_LENGTH = original_seq_len

        except Exception as e:
            print(f"  ERROR: {e}")
            config.SEQUENCE_LENGTH = original_seq_len
            continue

    save_path = config.EXPERIMENTS_DIR / "window_ablation_v2.csv"
    if results:
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to: {save_path}")

    if results:
        best = max(results, key=lambda r: r["test_acc"])
        fastest = max(results, key=lambda r: r["throughput_wps"])
        print(f"\n  ── Summary ──")
        print(f"  Best accuracy:    window={best['window_size']}, acc={best['test_acc']:.4f}")
        print(f"  Fastest:          window={fastest['window_size']}, "
              f"{fastest['throughput_wps']:,.0f} win/sec")

    print(f"\n{'=' * 70}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Window size ablation study")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sizes", nargs="+", type=int, default=WINDOW_SIZES)
    args = parser.parse_args()

    run_window_ablation(epochs_per_window=args.epochs, window_sizes=args.sizes)

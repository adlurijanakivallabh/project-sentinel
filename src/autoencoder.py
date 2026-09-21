import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src import clean_state_dict
from src.sessionization import load_sequences


class FlowAutoencoder(nn.Module):

    def __init__(self, num_features: int, latent_dim: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

    def get_reconstruction_error(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            mse = ((x - recon) ** 2).mean(dim=1)
        return mse


class FlowDataset(Dataset):
    def __init__(self, flows):
        self.flows = flows

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, idx):
        return self.flows[idx]


class FlowNormalizer:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, flows_tensor):
        flows_np = flows_tensor.numpy() if isinstance(flows_tensor, torch.Tensor) else flows_tensor
        self.mean = flows_np.mean(axis=0).astype(np.float32)
        self.std = flows_np.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, flows_tensor):
        if isinstance(flows_tensor, torch.Tensor):
            mean_t = torch.from_numpy(self.mean).to(flows_tensor.device)
            std_t = torch.from_numpy(self.std).to(flows_tensor.device)
            return (flows_tensor - mean_t) / std_t
        else:
            return (flows_tensor - self.mean) / self.std

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean, "std": self.std}, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.mean = data["mean"]
        obj.std = data["std"]
        return obj


def train_autoencoder(num_epochs=50, latent_dim=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  PHASE 6a: AUTOENCODER ZERO-DAY DETECTION")
    print("=" * 60)
    print(f"  Device: {device}")

    print("\n  Loading sequences...")
    seqs, lbls, _ = load_sequences()
    print(f"  Total sequences: {len(lbls):,}")

    normal_id = config.meta_label_to_id("Normal")
    normal_mask = lbls == normal_id
    normal_seqs = seqs[normal_mask]
    num_features = normal_seqs.shape[2]

    normal_flows = normal_seqs.reshape(-1, num_features)
    print(f"  Normal flows for training: {len(normal_flows):,}")
    print(f"  Feature dimensionality: {num_features}")

    print("\n  Fitting per-feature normalizer on Normal traffic...")
    normalizer = FlowNormalizer()
    normalizer.fit(normal_flows)

    scale_ratio = normalizer.std.max() / normalizer.std.min()
    print(f"  Feature std range: {normalizer.std.min():.6f} - {normalizer.std.max():.2f}")
    print(f"  Scale ratio (max/min std): {scale_ratio:.0f}x")
    print(f"  --> Normalization is ESSENTIAL (without it, 4 features dominate MSE)")

    normal_flows_norm = normalizer.transform(normal_flows)

    torch.manual_seed(config.RANDOM_STATE)
    n = len(normal_flows_norm)
    perm = torch.randperm(n)
    split = int(n * 0.8)
    train_flows = normal_flows_norm[perm[:split]]
    val_flows = normal_flows_norm[perm[split:]]

    train_ds = FlowDataset(train_flows)
    val_ds = FlowDataset(val_flows)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    print(f"  Train flows: {len(train_flows):,}")
    print(f"  Val flows (threshold calibration): {len(val_flows):,}")

    model = FlowAutoencoder(num_features=num_features, latent_dim=latent_dim)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")
    print(f"  Latent dimension: {latent_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    print(f"\n  {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}")
    print(f"  {'-' * 50}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
            train_n += len(batch)
        train_loss /= train_n

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item() * len(batch)
                val_n += len(batch)
        val_loss /= val_n

        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"  {epoch:5d}  {train_loss:12.6f}  {val_loss:12.6f}  {lr:10.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_features": num_features,
                "latent_dim": latent_dim,
                "val_loss": val_loss,
                "epoch": epoch,
            }
            save_path = config.MODELS_DIR / "autoencoder.pth"
            torch.save(checkpoint, save_path)

    print(f"\n  Best val loss: {best_val_loss:.6f}")
    print(f"  Model saved to: {save_path}")

    norm_path = config.MODELS_DIR / "autoencoder_normalizer.pkl"
    normalizer.save(norm_path)
    print(f"  Normalizer saved to: {norm_path}")

    print("\n  Calibrating anomaly threshold...")
    model.eval()
    val_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            errors = model.get_reconstruction_error(batch)
            val_errors.extend(errors.cpu().numpy())

    val_errors = np.array(val_errors)
    threshold_99 = float(np.percentile(val_errors, 99))
    threshold_95 = float(np.percentile(val_errors, 95))

    print(f"  Normal MSE -- mean: {val_errors.mean():.6f}, "
          f"std: {val_errors.std():.6f}")
    print(f"  Threshold (P95): {threshold_95:.6f}")
    print(f"  Threshold (P99): {threshold_99:.6f}  <-- USING THIS")

    threshold_data = {
        "threshold_p99": threshold_99,
        "threshold_p95": threshold_95,
        "normal_mse_mean": float(val_errors.mean()),
        "normal_mse_std": float(val_errors.std()),
    }
    torch.save(threshold_data, config.MODELS_DIR / "autoencoder_threshold.pth")

    return model, threshold_99, normalizer


def evaluate_autoencoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  AUTOENCODER EVALUATION")
    print("=" * 60)

    ae_path = config.MODELS_DIR / "autoencoder.pth"
    if not ae_path.exists():
        print("  ERROR: Autoencoder not trained yet. Run training first.")
        return

    checkpoint = torch.load(ae_path, map_location=device, weights_only=True)
    num_features = checkpoint["num_features"]
    latent_dim = checkpoint["latent_dim"]

    model = FlowAutoencoder(num_features=num_features, latent_dim=latent_dim)
    model.load_state_dict(clean_state_dict(checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    norm_path = config.MODELS_DIR / "autoencoder_normalizer.pkl"
    if not norm_path.exists():
        print("  ERROR: Normalizer not found. Retrain the autoencoder.")
        return
    normalizer = FlowNormalizer.load(norm_path)

    thresh_data = torch.load(
        config.MODELS_DIR / "autoencoder_threshold.pth",
        map_location="cpu", weights_only=True
    )
    threshold = thresh_data["threshold_p99"]

    seqs, lbls, _ = load_sequences()
    class_names = config.get_class_names()

    print(f"\n  Threshold (P99): {threshold:.6f}")
    print(f"\n  {'Class':<25}  {'N Flows':>10}  {'Mean MSE':>10}  "
          f"{'Flagged %':>10}  {'Status':>12}")
    print(f"  {'-' * 75}")

    results = {}
    total_flagged = 0
    total_attack_flows = 0

    for cls_id, cls_name in enumerate(class_names):
        mask = lbls == cls_id
        if mask.sum() == 0:
            continue

        cls_seqs = seqs[mask]
        cls_flows = cls_seqs.reshape(-1, num_features)

        cls_flows_norm = normalizer.transform(cls_flows).to(device)

        with torch.no_grad():
            errors = model.get_reconstruction_error(cls_flows_norm)
        errors_np = errors.cpu().numpy()

        flagged = (errors_np > threshold).sum()
        flagged_pct = flagged / len(errors_np) * 100
        mean_mse = errors_np.mean()

        if cls_id == 0:
            status = "[Normal]"
        elif flagged_pct > 50:
            status = "[DETECTED]"
        elif flagged_pct > 20:
            status = "[Partial]"
        else:
            status = "[Low]"

        results[cls_name] = {
            "n_flows": len(errors_np),
            "mean_mse": float(mean_mse),
            "flagged_pct": float(flagged_pct),
        }

        if cls_id > 0:
            total_flagged += flagged
            total_attack_flows += len(errors_np)

        print(f"  {cls_name:<25}  {len(errors_np):>10,}  {mean_mse:>10.6f}  "
              f"{flagged_pct:>9.1f}%  {status:>12}")

    if total_attack_flows > 0:
        overall_detection = total_flagged / total_attack_flows * 100
        print(f"\n  Overall attack detection rate: {overall_detection:.1f}%")
        print(f"  False positive rate (Normal flagged): "
              f"{results.get('Normal', {}).get('flagged_pct', 0):.1f}%")

    _plot_mse_distributions(model, seqs, lbls, class_names, threshold, device, normalizer)

    report_path = config.RESULTS_DIR / "autoencoder_report.txt"
    with open(report_path, "w") as f:
        f.write("Autoencoder Zero-Day Detection Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Threshold (P99): {threshold:.6f}\n\n")
        for cls_name, r in results.items():
            f.write(f"{cls_name}: MSE={r['mean_mse']:.6f}, Flagged={r['flagged_pct']:.1f}%\n")
        if total_attack_flows > 0:
            f.write(f"\nOverall attack detection: {total_flagged/total_attack_flows*100:.1f}%\n")
    print(f"  Report saved to {report_path}")

    print(f"\n{'=' * 60}")
    print("  AUTOENCODER EVALUATION COMPLETE")
    print(f"{'=' * 60}\n")

    return results


def _plot_mse_distributions(model, seqs, lbls, class_names, threshold, device, normalizer):
    fig, ax = plt.subplots(figsize=(12, 6))
    num_features = seqs.shape[2]

    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))

    for cls_id, cls_name in enumerate(class_names):
        mask = lbls == cls_id
        if mask.sum() == 0:
            continue

        cls_flows = seqs[mask].reshape(-1, num_features)
        if len(cls_flows) > 10000:
            idx = torch.randperm(len(cls_flows))[:10000]
            cls_flows = cls_flows[idx]

        cls_flows_norm = normalizer.transform(cls_flows).to(device)

        with torch.no_grad():
            errors = model.get_reconstruction_error(cls_flows_norm).cpu().numpy()

        ax.hist(errors, bins=100, alpha=0.5, label=cls_name,
                color=colors[cls_id], density=True)

    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"Threshold (P99): {threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Autoencoder MSE Distribution by Class (Normalized)", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, min(threshold * 5, ax.get_xlim()[1]))
    plt.tight_layout()

    path = config.FIGURES_DIR / "autoencoder_mse_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  MSE distribution plot saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder Zero-Day Detector")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate existing model")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--latent-dim", type=int, default=10,
                        help="Latent space dimension")
    args = parser.parse_args()

    if not args.eval_only:
        train_autoencoder(num_epochs=args.epochs, latent_dim=args.latent_dim)

    evaluate_autoencoder()

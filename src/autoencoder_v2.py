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


class WindowLSTMAutoencoder(nn.Module):

    def __init__(self, num_features: int = 30, seq_len: int = 20,
                 hidden_dim: int = 64, latent_dim: int = 16, num_layers: int = 1):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.decoder_output = nn.Linear(hidden_dim, num_features)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]
        z = self.encoder_fc(h_last)
        return z

    def decode(self, z):
        h = self.decoder_fc(z)
        h_repeated = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder_lstm(h_repeated)
        recon = self.decoder_output(decoded)
        return recon

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def get_reconstruction_error(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            mse = ((x - recon) ** 2).mean(dim=(1, 2))
        return mse


class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


class WindowNormalizer:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, windows_tensor):
        flat = windows_tensor.reshape(-1, windows_tensor.shape[-1]).numpy()
        self.mean = flat.mean(axis=0).astype(np.float32)
        self.std = flat.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, windows_tensor):
        mean_t = torch.from_numpy(self.mean).to(windows_tensor.device)
        std_t = torch.from_numpy(self.std).to(windows_tensor.device)
        return (windows_tensor - mean_t) / std_t

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


def train_window_autoencoder(num_epochs=80, latent_dim=16, hidden_dim=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  AUTOENCODER V2: WINDOW-LEVEL LSTM AUTOENCODER")
    print("=" * 60)
    print(f"  Device: {device}")

    print("\n  Loading sequences...")
    seqs, lbls, _ = load_sequences()
    num_features = seqs.shape[2]
    seq_len = seqs.shape[1]
    print(f"  Total windows: {len(lbls):,}")
    print(f"  Window shape: ({seq_len}, {num_features})")

    normal_id = config.meta_label_to_id("Normal")
    normal_mask = lbls == normal_id
    normal_windows = seqs[normal_mask]
    print(f"  Normal windows for training: {len(normal_windows):,}")

    print("\n  Fitting per-feature normalizer on Normal windows...")
    normalizer = WindowNormalizer()
    normalizer.fit(normal_windows)
    normal_windows_norm = normalizer.transform(normal_windows)

    torch.manual_seed(config.RANDOM_STATE)
    n = len(normal_windows_norm)
    perm = torch.randperm(n)
    split = int(n * 0.8)
    train_windows = normal_windows_norm[perm[:split]]
    val_windows = normal_windows_norm[perm[split:]]

    train_ds = WindowDataset(train_windows)
    val_ds = WindowDataset(val_windows)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    print(f"  Train windows: {len(train_windows):,}")
    print(f"  Val windows: {len(val_windows):,}")

    model = WindowLSTMAutoencoder(
        num_features=num_features,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")
    print(f"  Hidden dim: {hidden_dim}, Latent dim: {latent_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        if epoch % 10 == 0 or epoch == num_epochs or epoch <= 5:
            print(f"  {epoch:5d}  {train_loss:12.6f}  {val_loss:12.6f}  {lr:10.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_features": num_features,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "val_loss": val_loss,
                "epoch": epoch,
            }
            save_path = config.MODELS_DIR / "autoencoder_v2.pth"
            torch.save(checkpoint, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n  Best val loss: {best_val_loss:.6f}")
    print(f"  Model saved to: {save_path}")

    norm_path = config.MODELS_DIR / "autoencoder_v2_normalizer.pkl"
    normalizer.save(norm_path)

    print("\n  Calibrating anomaly threshold...")
    best_ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(clean_state_dict(best_ckpt["model_state_dict"]))
    model.eval()

    val_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            errors = model.get_reconstruction_error(batch)
            val_errors.extend(errors.cpu().numpy())

    val_errors = np.array(val_errors)
    threshold_p99 = float(np.percentile(val_errors, 99))
    threshold_p97 = float(np.percentile(val_errors, 97))
    threshold_p95 = float(np.percentile(val_errors, 95))

    print(f"  Normal MSE -- mean: {val_errors.mean():.6f}, std: {val_errors.std():.6f}")
    print(f"  Threshold (P95): {threshold_p95:.6f}")
    print(f"  Threshold (P97): {threshold_p97:.6f}")
    print(f"  Threshold (P99): {threshold_p99:.6f}")

    threshold_data = {
        "threshold_p99": threshold_p99,
        "threshold_p97": threshold_p97,
        "threshold_p95": threshold_p95,
        "normal_mse_mean": float(val_errors.mean()),
        "normal_mse_std": float(val_errors.std()),
    }
    torch.save(threshold_data, config.MODELS_DIR / "autoencoder_v2_threshold.pth")

    return model, threshold_data, normalizer


def evaluate_window_autoencoder(threshold_key="threshold_p99"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  AUTOENCODER V2 EVALUATION (Window-Level LSTM)")
    print("=" * 60)

    ae_path = config.MODELS_DIR / "autoencoder_v2.pth"
    if not ae_path.exists():
        print("  ERROR: Autoencoder V2 not trained yet.")
        return
    checkpoint = torch.load(ae_path, map_location=device, weights_only=False)

    model = WindowLSTMAutoencoder(
        num_features=checkpoint["num_features"],
        seq_len=checkpoint["seq_len"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
    )
    model.load_state_dict(clean_state_dict(checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    normalizer = WindowNormalizer.load(config.MODELS_DIR / "autoencoder_v2_normalizer.pkl")
    thresh_data = torch.load(
        config.MODELS_DIR / "autoencoder_v2_threshold.pth",
        map_location="cpu", weights_only=True
    )

    seqs, lbls, _ = load_sequences()
    class_names = config.get_class_names()

    for tkey in ["threshold_p99", "threshold_p97", "threshold_p95"]:
        threshold = thresh_data[tkey]
        print(f"\n  --- Results with {tkey} = {threshold:.6f} ---")
        print(f"  {'Class':<25}  {'N Windows':>10}  {'Mean MSE':>10}  {'Flagged %':>10}")
        print(f"  {'-' * 65}")

        total_flagged = 0
        total_attack = 0
        results = {}

        for cls_id, cls_name in enumerate(class_names):
            mask = lbls == cls_id
            if mask.sum() == 0:
                continue

            cls_windows = normalizer.transform(seqs[mask]).to(device)
            n_windows = len(cls_windows)

            all_errors = []
            with torch.no_grad():
                for i in range(0, n_windows, 256):
                    batch = cls_windows[i:i+256]
                    errors = model.get_reconstruction_error(batch)
                    all_errors.extend(errors.cpu().numpy())
            all_errors = np.array(all_errors)

            flagged = (all_errors > threshold).sum()
            flagged_pct = flagged / len(all_errors) * 100
            mean_mse = all_errors.mean()

            results[cls_name] = {
                "n_windows": n_windows,
                "mean_mse": float(mean_mse),
                "flagged_pct": float(flagged_pct),
            }

            if cls_id > 0:
                total_flagged += flagged
                total_attack += n_windows

            print(f"  {cls_name:<25}  {n_windows:>10,}  {mean_mse:>10.6f}  {flagged_pct:>9.1f}%")

        if total_attack > 0:
            overall = total_flagged / total_attack * 100
            fp_rate = results.get("Normal", {}).get("flagged_pct", 0)
            print(f"\n  Overall attack detection: {overall:.1f}%")
            print(f"  False positive rate: {fp_rate:.1f}%")

    threshold = thresh_data[threshold_key]
    report_path = config.RESULTS_DIR / "autoencoder_v2_report.txt"
    with open(report_path, "w") as f:
        f.write("Autoencoder V2 (Window-Level LSTM) Detection Results\n")
        f.write("=" * 55 + "\n")
        f.write(f"Architecture: LSTM Encoder-Decoder (window-level)\n")
        f.write(f"Threshold ({threshold_key}): {threshold:.6f}\n\n")

        total_flagged = 0
        total_attack = 0
        for cls_id, cls_name in enumerate(class_names):
            mask = lbls == cls_id
            if mask.sum() == 0:
                continue
            cls_windows = normalizer.transform(seqs[mask]).to(device)
            all_errors = []
            with torch.no_grad():
                for i in range(0, len(cls_windows), 256):
                    errors = model.get_reconstruction_error(cls_windows[i:i+256])
                    all_errors.extend(errors.cpu().numpy())
            all_errors = np.array(all_errors)
            flagged = (all_errors > threshold).sum()
            flagged_pct = flagged / len(all_errors) * 100
            f.write(f"{cls_name}: MSE={all_errors.mean():.6f}, Flagged={flagged_pct:.1f}%\n")
            if cls_id > 0:
                total_flagged += flagged
                total_attack += len(all_errors)
        if total_attack > 0:
            f.write(f"\nOverall attack detection: {total_flagged/total_attack*100:.1f}%\n")

    print(f"\n  Report saved to {report_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder V2 — Window-Level LSTM")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    if not args.eval_only:
        train_window_autoencoder(
            num_epochs=args.epochs,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        )

    evaluate_window_autoencoder()

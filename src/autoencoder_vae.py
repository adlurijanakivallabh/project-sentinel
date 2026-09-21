import sys
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config


class LSTMVariationalAutoencoder(nn.Module):
    def __init__(self, input_size=30, seq_len=20, hidden_size=128,
                 latent_dim=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, batch_first=True,
            num_layers=num_layers, dropout=dropout,
        )

        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        self.decoder_init = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            hidden_size, hidden_size, batch_first=True,
            num_layers=num_layers, dropout=dropout,
        )
        self.decoder_out = nn.Linear(hidden_size, input_size)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x):
        _, (h, _) = self.encoder_lstm(x)
        h_final = h[-1]
        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)
        return mu, logvar

    def decode(self, z):
        dec_init = self.decoder_init(z)
        dec_input = dec_init.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_input)
        recon = self.decoder_out(dec_out)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def anomaly_score(self, x):
        recon, mu, logvar = self.forward(x)

        recon_loss = F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2])

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)

        return recon_loss + 0.1 * kl_loss


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def train_vae(epochs=100, lr=1e-3, beta=1.0, batch_size=128):
    from src.sessionization import load_sequences
    from src.split_utils import block_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"  TRAINING LSTM-VAE (Zero-Day Detector)")
    print(f"  Device: {device} | beta={beta}")
    print(f"{'=' * 60}\n")

    seqs, lbls, _ = load_sequences()

    seq_flat = seqs.reshape(-1, seqs.shape[-1])
    mean = seq_flat.mean(dim=0, keepdim=True)
    std = seq_flat.std(dim=0, keepdim=True).clamp(min=1e-8)
    seqs = ((seqs - mean) / std).clamp(-10, 10)

    n = len(lbls)
    shuffle_idx = torch.randperm(n, generator=torch.Generator().manual_seed(config.RANDOM_STATE))
    seqs = seqs[shuffle_idx]
    lbls = lbls[shuffle_idx]
    train_idx, val_idx, test_idx = block_split(n)

    normal_id = config.meta_label_to_id("Normal")

    train_mask = lbls[train_idx] == normal_id
    val_mask_normal = lbls[val_idx] == normal_id
    val_mask_attack = lbls[val_idx] != normal_id

    train_normal = seqs[train_idx][train_mask]
    val_normal = seqs[val_idx][val_mask_normal]
    val_attack = seqs[val_idx][val_mask_attack]

    print(f"  Train (Normal):      {len(train_normal):,}")
    print(f"  Val (Normal):        {len(val_normal):,}")
    print(f"  Val (Attack):        {len(val_attack):,}")

    train_loader = DataLoader(
        TensorDataset(train_normal),
        batch_size=batch_size, shuffle=True,
    )

    n_features = seqs.shape[2]
    seq_len = seqs.shape[1]
    model = LSTMVariationalAutoencoder(
        input_size=n_features,
        seq_len=seq_len,
    )
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    save_path = config.MODELS_DIR / "autoencoder_vae.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        count = 0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            count += x.size(0)

        train_loss /= count

        model.eval()
        with torch.no_grad():
            normal_scores = model.anomaly_score(val_normal.to(device))
            attack_scores = model.anomaly_score(val_attack[:1000].to(device))

            normal_mean = normal_scores.mean().item()
            attack_mean = attack_scores.mean().item()
            separation = attack_mean / max(normal_mean, 1e-8)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:.6f} | "
                  f"Normal: {normal_mean:.4f} | Attack: {attack_mean:.4f} | "
                  f"Sep: {separation:.2f}x")

        if train_loss < best_val_loss:
            best_val_loss = train_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": n_features,
                "seq_len": seq_len,
                "epoch": epoch,
                "loss": train_loss,
            }, save_path)

        scheduler.step()

    model.eval()
    with torch.no_grad():
        test_seqs = seqs[test_idx].to(device)
        test_lbls_np = lbls[test_idx].numpy()
        test_scores = []
        for i in range(0, len(test_seqs), batch_size):
            batch = test_seqs[i:i + batch_size]
            scores = model.anomaly_score(batch)
            test_scores.append(scores.cpu())
        test_scores = torch.cat(test_scores).numpy()

    normal_scores_test = test_scores[test_lbls_np == normal_id]
    threshold = float(np.percentile(normal_scores_test, 95))

    print(f"\n  -- Detection Rates (threshold={threshold:.4f}) --")
    class_names = config.get_class_names()
    for cls_id, cls_name in enumerate(class_names):
        mask = test_lbls_np == cls_id
        if mask.sum() == 0:
            continue
        detected = (test_scores[mask] > threshold).sum()
        rate = detected / mask.sum() * 100
        print(f"    {cls_name:25s}: {rate:6.1f}% ({detected}/{mask.sum()})")

    print(f"\n  Saved to: {save_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train LSTM-VAE for zero-day detection")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_vae(epochs=args.epochs, beta=args.beta, lr=args.lr)

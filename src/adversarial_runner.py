import sys
from pathlib import Path
import torch
import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src import clean_state_dict
from src.sessionization import load_sequences
from src.train import get_model, SequenceDataset
from src.split_utils import block_split
from torch.utils.data import DataLoader
import torch.nn as nn


def fgsm_attack(model, x, y, epsilon, criterion):
    x_adv = x.clone().detach().requires_grad_(True)
    out = model(x_adv)
    loss = criterion(out, y)
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return x_adv.detach()


def evaluate_clean(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def evaluate_gaussian(model, loader, device, sigma):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_noisy = x + torch.randn_like(x) * sigma
            pred = model(x_noisy).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def evaluate_fgsm(model, loader, device, epsilon):
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = fgsm_attack(model, x, y, epsilon, criterion)
        model.eval()
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        model.train()
        correct += (pred == y).sum().item()
        total += y.size(0)
    model.eval()
    return correct / total


def run_adversarial_experiments():
    print("\n" + "=" * 60)
    print("  ADVERSARIAL ROBUSTNESS TESTING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    test_ds = SequenceDataset(seqs[test_idx], lbls[test_idx])
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    print(f"\n  Model: {variant} ({num_features} features, {num_classes} classes)")
    model = get_model(variant, num_features=num_features, num_classes=num_classes)
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    model.to(device).eval()

    clean_acc = evaluate_clean(model, test_loader, device)
    print(f"  Clean accuracy: {clean_acc*100:.2f}%")

    print(f"\n  --- Gaussian Noise Robustness ---")
    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
    gauss_accs = []
    for sigma in noise_levels:
        acc = evaluate_gaussian(model, test_loader, device, sigma)
        gauss_accs.append(acc)
        drop = (clean_acc - acc) * 100
        print(f"    sigma={sigma:.2f}: {acc*100:.2f}% (drop: {drop:+.2f}%)")

    print(f"\n  --- FGSM Attack Robustness ---")
    epsilons = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    fgsm_accs = []
    for eps in epsilons:
        acc = evaluate_fgsm(model, test_loader, device, eps)
        fgsm_accs.append(acc)
        drop = (clean_acc - acc) * 100
        print(f"    epsilon={eps:.2f}: {acc*100:.2f}% (drop: {drop:+.2f}%)")

    print(f"\n{'=' * 60}")
    print(f"  ADVERSARIAL ROBUSTNESS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Attack':<25} {'Strength':>10} {'Accuracy':>10} {'Drop':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Clean (no attack)':<25} {'---':>10} {clean_acc*100:>9.2f}% {'---':>8}")
    for i, sigma in enumerate(noise_levels[1:], 1):
        print(f"  {f'Gaussian (sigma={sigma})':<25} {sigma:>10.2f} {gauss_accs[i]*100:>9.2f}% {(clean_acc-gauss_accs[i])*100:>7.2f}%")
    for i, eps in enumerate(epsilons[1:], 1):
        print(f"  {f'FGSM (eps={eps})':<25} {eps:>10.2f} {fgsm_accs[i]*100:>9.2f}% {(clean_acc-fgsm_accs[i])*100:>7.2f}%")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run_adversarial_experiments()

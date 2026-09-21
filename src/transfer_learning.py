import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src import clean_state_dict
from src.train import get_model, SequenceDataset
from src.cross_dataset_eval import load_unsw_nb15, create_windows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("  G: TRANSFER LEARNING ON UNSW-NB15")
    print("=" * 60)

    features, labels = load_unsw_nb15()
    unsw_seqs, unsw_lbls = create_windows(features, labels, window_size=config.SEQUENCE_LENGTH)
    print(f"  UNSW-NB15 windows: {len(unsw_lbls):,}")
    print(f"  Shape: {tuple(unsw_seqs.shape)}")

    num_model_features = 30
    if unsw_seqs.shape[2] < num_model_features:
        pad = num_model_features - unsw_seqs.shape[2]
        unsw_seqs = torch.cat([unsw_seqs, torch.zeros(unsw_seqs.shape[0], unsw_seqs.shape[1], pad)], dim=2)
        print(f"  Padded to {num_model_features} features ({pad} temporal zeros)")

    names = config.get_class_names()
    for i, name in enumerate(names):
        cnt = (unsw_lbls == i).sum().item()
        if cnt > 0:
            print(f"    {name}: {cnt}")

    print("\n  --- Zero-shot (no fine-tuning) ---")
    ckpt = torch.load(config.MODEL_CHECKPOINT_PATH, map_location=device, weights_only=False)
    variant = ckpt.get("model_variant", "cnn_lstm_attention")
    num_features = ckpt.get("num_features", num_model_features)
    num_classes = ckpt.get("num_classes", config.NUM_CLASSES)
    print(f"  Model variant: {variant} ({num_features} features, {num_classes} classes)")
    model = get_model(variant, num_features=num_features, num_classes=num_classes)
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    model.to(device).eval()

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(unsw_seqs), 256):
            batch = unsw_seqs[i:i+256].to(device)
            preds = model(batch).argmax(1).cpu().tolist()
            all_preds.extend(preds)
    all_preds = np.array(all_preds)
    unsw_np = unsw_lbls.numpy()
    zero_shot_acc = (all_preds == unsw_np).mean() * 100
    print(f"  Zero-shot accuracy: {zero_shot_acc:.2f}%")

    torch.manual_seed(42)
    n = len(unsw_lbls)
    perm = torch.randperm(n)
    split = n // 2
    train_idx = perm[:split]
    test_idx = perm[split:]

    train_ds = SequenceDataset(unsw_seqs[train_idx], unsw_lbls[train_idx])
    test_ds = SequenceDataset(unsw_seqs[test_idx], unsw_lbls[test_idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    print(f"\n  Train: {len(train_idx):,}, Test: {len(test_idx):,}")

    model_ft = get_model(variant, num_features=num_features, num_classes=num_classes)
    model_ft.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    model_ft.to(device)

    for name, param in model_ft.named_parameters():
        if "attention" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_ft.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    from collections import Counter
    counts = Counter(unsw_lbls[train_idx].numpy().tolist())
    weights = torch.zeros(config.NUM_CLASSES)
    for c in range(config.NUM_CLASSES):
        weights[c] = 1.0 / max(counts.get(c, 1), 1)
    weights = weights / weights.sum() * config.NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_ft.parameters()),
        lr=5e-4, weight_decay=1e-5,
    )

    print("\n  Fine-tuning (head only)...")
    for epoch in range(1, 21):
        model_ft.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model_ft(X), y)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            model_ft.eval()
            correct = 0
            total_n = 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    preds = model_ft(X).argmax(1)
                    correct += (preds == y).sum().item()
                    total_n += len(y)
            acc = correct / total_n * 100
            print(f"    Epoch {epoch}: test_acc={acc:.2f}%")

    model_ft.eval()
    ft_preds = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = model_ft(X).argmax(1).cpu().tolist()
            ft_preds.extend(preds)
    ft_preds = np.array(ft_preds)
    test_labels = unsw_lbls[test_idx].numpy()
    ft_acc = (ft_preds == test_labels).mean() * 100

    from sklearn.metrics import classification_report
    print(f"\n  --- Fine-tuned Results ---")
    print(f"  Accuracy: {ft_acc:.2f}%")
    all_labels = list(range(num_classes))
    print(classification_report(test_labels, ft_preds, labels=all_labels, target_names=names, digits=4, zero_division=0))

    print("\n  --- Full Fine-tuning (all layers) ---")
    model_full = get_model(variant, num_features=num_features, num_classes=num_classes)
    model_full.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    model_full.to(device)
    for param in model_full.parameters():
        param.requires_grad = True

    optimizer_full = torch.optim.Adam(model_full.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(1, 21):
        model_full.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer_full.zero_grad()
            loss = criterion(model_full(X), y)
            loss.backward()
            optimizer_full.step()

        if epoch % 5 == 0:
            model_full.eval()
            correct = 0
            total_n = 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    preds = model_full(X).argmax(1)
                    correct += (preds == y).sum().item()
                    total_n += len(y)
            acc = correct / total_n * 100
            print(f"    Epoch {epoch}: test_acc={acc:.2f}%")

    model_full.eval()
    full_preds = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = model_full(X).argmax(1).cpu().tolist()
            full_preds.extend(preds)
    full_preds = np.array(full_preds)
    full_acc = (full_preds == test_labels).mean() * 100
    print(f"\n  Full fine-tuning accuracy: {full_acc:.2f}%")
    print(classification_report(test_labels, full_preds, labels=all_labels, target_names=names, digits=4, zero_division=0))

    print(f"\n  === SUMMARY ===")
    print(f"  Zero-shot (CIC-IDS2017 -> UNSW-NB15): {zero_shot_acc:.2f}%")
    print(f"  Head-only fine-tune (20 epochs):      {ft_acc:.2f}%")
    print(f"  Full fine-tune (20 epochs):           {full_acc:.2f}%")
    print(f"  Improvement: +{full_acc - zero_shot_acc:.2f}%")

    print(f"\n{'=' * 60}")
    print("  TRANSFER LEARNING COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

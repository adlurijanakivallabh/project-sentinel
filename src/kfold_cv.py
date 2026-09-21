import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.sessionization import load_sequences
from src.train import get_model, SequenceDataset, _post_split_oversample
from src.losses import get_loss_function
from sklearn.metrics import matthews_corrcoef, f1_score


def block_kfold_split(n, k=3, block_size=10):
    n_blocks = n // block_size
    remainder = n % block_size
    
    block_indices = np.arange(n_blocks)
    np.random.seed(42)
    np.random.shuffle(block_indices)
    
    fold_size = n_blocks // k
    folds = []
    
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_blocks
        test_blocks = block_indices[start:end]
        train_blocks = np.setdiff1d(block_indices, test_blocks)
        
        test_idx = []
        for b in test_blocks:
            test_idx.extend(range(b * block_size, min((b + 1) * block_size, n)))
        
        train_idx = []
        for b in train_blocks:
            train_idx.extend(range(b * block_size, min((b + 1) * block_size, n)))
        
        if remainder > 0:
            train_idx.extend(range(n_blocks * block_size, n))
        
        folds.append((np.array(train_idx), np.array(test_idx)))
    
    return folds


def train_fold(seqs, lbls, train_idx, test_idx, fold_num, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()
    
    train_seqs_raw = seqs[train_idx]
    train_flat = train_seqs_raw.reshape(-1, train_seqs_raw.shape[-1])
    zscore_mean = train_flat.mean(dim=0, keepdim=True)
    zscore_std = train_flat.std(dim=0, keepdim=True).clamp(min=1e-8)
    
    seqs_norm = ((seqs - zscore_mean) / zscore_std).clamp(-10, 10)
    
    train_seqs = seqs_norm[train_idx]
    train_lbls = lbls[train_idx]
    test_seqs = seqs_norm[test_idx]
    test_lbls_np = lbls[test_idx].numpy()
    
    train_seqs, train_lbls = _post_split_oversample(train_seqs, train_lbls)
    
    class_counts = torch.bincount(train_lbls, minlength=config.NUM_CLASSES).float()
    class_weights = torch.zeros_like(class_counts)
    active_mask = class_counts > 0
    if active_mask.any():
        inv_freq = 1.0 / class_counts[active_mask].sqrt()
        inv_freq = inv_freq / inv_freq.sum() * active_mask.sum().float()
        class_weights[active_mask] = inv_freq
    class_weights = class_weights.to(device)
    
    model = get_model("model_v2", num_features=seqs.shape[2])
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.V2_LEARNING_RATE, 
                                   weight_decay=config.V2_WEIGHT_DECAY)
    criterion = get_loss_function(config, class_weights=class_weights)
    scaler = GradScaler('cuda', enabled=use_amp)
    
    batch_size = config.V2_BATCH_SIZE
    train_ds = SequenceDataset(train_seqs, train_lbls)
    test_ds = SequenceDataset(test_seqs, lbls[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    
    print(f"    Training {len(train_ds):,} samples, {num_epochs} epochs...", flush=True)
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.V2_GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        
        train_acc = correct / total
        train_loss = running_loss / total
        print(f"      Epoch {epoch:2d}/{num_epochs}: loss={train_loss:.4f}, acc={train_acc:.4f}", flush=True)
    
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            with autocast('cuda', enabled=use_amp):
                out = model(x)
            all_preds.append(out.argmax(1).cpu().numpy())
    
    preds = np.concatenate(all_preds)
    accuracy = (preds == test_lbls_np).mean()
    mcc = matthews_corrcoef(test_lbls_np, preds)
    macro_f1 = f1_score(test_lbls_np, preds, average='macro', zero_division=0)
    
    print(f"    >> Fold {fold_num} Result: Accuracy={accuracy:.4f}, MCC={mcc:.4f}, Macro-F1={macro_f1:.4f}", flush=True)
    
    return {"accuracy": accuracy, "mcc": mcc, "macro_f1": macro_f1}


def run_kfold(k=3, epochs=10):
    print(f"\n{'='*70}")
    print(f"  {k}-FOLD BLOCK-BASED CROSS-VALIDATION")
    print(f"  Model: MultiScale CNN-BiLSTM-GRU-MHA (2.87M params)")
    print(f"  Epochs per fold: {epochs}")
    print(f"{'='*70}\n", flush=True)
    
    seqs, lbls, _ = load_sequences()
    print(f"  Data: {tuple(seqs.shape)}", flush=True)
    
    folds = block_kfold_split(len(lbls), k=k, block_size=10)
    
    results = []
    total_start = time.time()
    
    for i, (train_idx, test_idx) in enumerate(folds):
        fold_start = time.time()
        print(f"\n  --- Fold {i+1}/{k} (train={len(train_idx):,}, test={len(test_idx):,}) ---", flush=True)
        fold_result = train_fold(seqs, lbls, train_idx, test_idx, i+1, epochs)
        fold_time = time.time() - fold_start
        print(f"    Fold {i+1} completed in {fold_time:.0f}s", flush=True)
        results.append(fold_result)
    
    total_time = time.time() - total_start
    
    accs = [r["accuracy"] for r in results]
    mccs = [r["mcc"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    
    print(f"\n{'='*70}")
    print(f"  CROSS-VALIDATION RESULTS ({k} folds, {total_time:.0f}s total)")
    print(f"{'='*70}")
    print(f"  Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  MCC:       {np.mean(mccs):.4f} +/- {np.std(mccs):.4f}")
    print(f"  Macro-F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"{'='*70}\n")
    
    for i, r in enumerate(results):
        print(f"  Fold {i+1}: Acc={r['accuracy']:.4f}, MCC={r['mcc']:.4f}, F1={r['macro_f1']:.4f}")
    
    print(flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Fold Cross-Validation")
    parser.add_argument("--folds", type=int, default=3, help="Number of folds (default: 3)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per fold (default: 10)")
    args = parser.parse_args()
    
    run_kfold(k=args.folds, epochs=args.epochs)

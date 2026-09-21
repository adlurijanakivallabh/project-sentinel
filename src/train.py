import sys
import argparse
import time
import copy
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.sessionization import load_sequences
from src.losses import get_loss_function
from src.split_utils import block_split


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


ALL_VARIANTS = [
    "model_v2", "cnn_lstm", "cnn_lstm_attention", "cnn_bilstm", "cnn_transformer"
]


def get_model(variant: str, num_features: int, num_classes: int = None):
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    if variant == "model_v2":
        from src.model_v2 import build_model_v2
        return build_model_v2(num_features=num_features, num_classes=num_classes)

    elif variant in ("cnn_lstm", "cnn_lstm_attention"):
        from src.model_cnn_lstm import CNNLSTM
        use_attention = (variant == "cnn_lstm_attention")
        return CNNLSTM(
            num_features=num_features,
            num_classes=num_classes,
            use_attention=use_attention,
        )

    elif variant == "cnn_bilstm":
        from src.model_cnn_bilstm import build_bilstm_model
        return build_bilstm_model(num_features=num_features, num_classes=num_classes)

    elif variant == "cnn_transformer":
        from src.model_cnn_transformer import build_transformer_model
        return build_transformer_model(num_features=num_features, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model variant: {variant}. Options: {ALL_VARIANTS}")


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
    
    def state_dict(self):
        return self.shadow.state_dict()


def add_adversarial_samples(seqs, lbls, noise_std=0.05, ratio=0.10):
    normal_id = config.meta_label_to_id("Normal")
    attack_mask = lbls != normal_id
    attack_seqs = seqs[attack_mask]
    attack_lbls = lbls[attack_mask]

    n_adv = int(len(attack_seqs) * ratio)
    if n_adv == 0:
        return seqs, lbls

    indices = torch.randperm(len(attack_seqs))[:n_adv]
    adv_seqs = attack_seqs[indices].clone()
    adv_lbls = attack_lbls[indices].clone()

    noise = torch.randn_like(adv_seqs) * noise_std
    adv_seqs = adv_seqs + noise

    print(f"[train] Added {n_adv:,} adversarial samples (std={noise_std})")

    return torch.cat([seqs, adv_seqs]), torch.cat([lbls, adv_lbls])


def _post_split_oversample(train_seqs, train_lbls):
    counts = torch.bincount(train_lbls, minlength=config.NUM_CLASSES)
    active_counts = counts[counts > 0]
    
    if len(active_counts) <= 1:
        return train_seqs, train_lbls
    
    MAX_DUPLICATION_RATIO = 50
    max_class = int(active_counts.max().item())
    
    extra_seqs = []
    extra_lbls = []
    
    for c in range(config.NUM_CLASSES):
        c_count = counts[c].item()
        if c_count == 0 or c_count >= max_class:
            continue
        
        class_target = min(max_class, c_count * MAX_DUPLICATION_RATIO)
        deficit = class_target - c_count
        if deficit <= 0:
            continue
        
        c_mask = train_lbls == c
        c_seqs = train_seqs[c_mask]
        
        indices = torch.randint(0, c_count, (deficit,))
        oversampled = c_seqs[indices].clone()
        
        duplication_ratio = deficit / max(c_count, 1)
        if duplication_ratio > 500:
            noise_sigma = 0.15
        elif duplication_ratio > 100:
            noise_sigma = 0.08
        elif duplication_ratio > 10:
            noise_sigma = 0.05
        else:
            noise_sigma = 0.0
        
        aug_tags = []
        
        if noise_sigma > 0:
            noise = torch.randn_like(oversampled) * noise_sigma
            oversampled = oversampled + noise
            aug_tags.append(f"noise={noise_sigma}")
        
        if duplication_ratio > 5:
            T = oversampled.size(1)
            for i in range(oversampled.size(0)):
                shift = np.random.randint(-2, 3)
                if shift != 0:
                    oversampled[i] = torch.roll(oversampled[i], shifts=shift, dims=0)
                    if shift > 0:
                        oversampled[i, :shift, :] = 0.0
                    else:
                        oversampled[i, shift:, :] = 0.0
            aug_tags.append("jitter+/-2")
        
        extra_seqs.append(oversampled)
        extra_lbls.append(torch.full((deficit,), c, dtype=train_lbls.dtype))
        
        class_names = config.get_class_names()
        name = class_names[c] if c < len(class_names) else str(c)
        tag_str = f" [{', '.join(aug_tags)}, {duplication_ratio:.0f}x dup]" if aug_tags else ""
        print(f"  Oversample {name}: {c_count:,} -> {class_target:,} (+{deficit:,}){tag_str}")
    
    if extra_seqs:
        train_seqs = torch.cat([train_seqs] + extra_seqs)
        train_lbls = torch.cat([train_lbls] + extra_lbls)
        print(f"  Post-split oversampled: {len(train_lbls):,} training windows (was {counts.sum().item():,})")
        print(f"  NOTE: Max duplication capped at {MAX_DUPLICATION_RATIO}x. Focal Loss class weights handle remaining imbalance.")
    else:
        print(f"  No oversampling needed (all classes >= {max_class:,})")
    
    return train_seqs, train_lbls


def train(model_variant: str = None, use_focal: bool = None):
    if model_variant is None:
        model_variant = config.MODEL_VARIANT
    if use_focal is None:
        use_focal = config.USE_FOCAL_LOSS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_STATE)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

    is_v2_model = (model_variant == "model_v2")
    batch_size = config.V2_BATCH_SIZE if is_v2_model else config.BATCH_SIZE
    num_epochs = config.V2_NUM_EPOCHS if is_v2_model else config.NUM_EPOCHS
    lr = config.V2_LEARNING_RATE if is_v2_model else config.LEARNING_RATE
    wd = config.V2_WEIGHT_DECAY if is_v2_model else config.WEIGHT_DECAY
    grad_clip = config.V2_GRAD_CLIP_NORM if is_v2_model else 1.0
    label_smooth = config.V2_LABEL_SMOOTHING if is_v2_model else config.LABEL_SMOOTHING
    swa_start = config.V2_SWA_START_EPOCH if is_v2_model else max(num_epochs - 5, 1)
    patience = config.V2_EARLY_STOPPING_PATIENCE if is_v2_model else config.EARLY_STOPPING_PATIENCE

    print(f"\n{'=' * 70}")
    if is_v2_model:
        print(f"  TRAINING - SentinelV2 (PRIMARY MODEL)")
    else:
        print(f"  TRAINING - {model_variant.upper()} (SECONDARY MODEL)")
    print(f"{'=' * 70}")
    print(f"  Device:          {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU:             {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  Loss:            {'Focal' if use_focal else 'CrossEntropy'}")
    print(f"  Label smoothing: {label_smooth}")
    print(f"  Epochs:          {num_epochs}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Learning rate:   {lr}")
    print(f"  AMP:             {'Enabled (FP16)' if use_amp else 'Disabled'}")
    print(f"  SWA:             Epoch {swa_start}+")
    print(f"  Grad clipping:   {grad_clip}")
    print(f"  Early stopping:  patience={patience}")

    seqs, lbls, _ = load_sequences()
    n = len(lbls)

    print(f"\n  Data shape:      {tuple(seqs.shape)}")
    print(f"  Classes:         {config.NUM_CLASSES}")

    train_idx, val_idx, test_idx = block_split(n)
    print(f"  Split:           {len(train_idx):,} train / {len(val_idx):,} val / {len(test_idx):,} test (block-split)")

    train_seqs_raw = seqs[train_idx]
    train_flat = train_seqs_raw.reshape(-1, train_seqs_raw.shape[-1])
    zscore_mean = train_flat.mean(dim=0, keepdim=True)
    zscore_std = train_flat.std(dim=0, keepdim=True).clamp(min=1e-8)

    seqs = ((seqs - zscore_mean) / zscore_std).clamp(-10, 10)
    print(f"  Z-score:         TRAIN-ONLY stats (no test leakage)")
    print(f"  Data range:      [{seqs.min():.2f}, {seqs.max():.2f}] (z-score normalized)")

    train_seqs, train_lbls = seqs[train_idx], lbls[train_idx]
    val_seqs, val_lbls = seqs[val_idx], lbls[val_idx]
    test_lbls = lbls[test_idx]

    class_names = config.get_class_names()
    train_classes = torch.unique(train_lbls)
    val_classes = torch.unique(val_lbls)
    test_classes = torch.unique(test_lbls)
    print(f"  Train classes:   {len(train_classes)}/{config.NUM_CLASSES} present")
    print(f"  Val classes:     {len(val_classes)}/{config.NUM_CLASSES} present")
    print(f"  Test classes:    {len(test_classes)}/{config.NUM_CLASSES} present")

    for partition_name, partition_lbls in [("Val", val_lbls), ("Test", test_lbls)]:
        counts = torch.bincount(partition_lbls, minlength=config.NUM_CLASSES)
        for c in range(config.NUM_CLASSES):
            if counts[c] < 5:
                name = class_names[c] if c < len(class_names) else str(c)
                print(f"  WARNING: {partition_name} has only {counts[c]} samples of '{name}' — low statistical power")

    train_seqs, train_lbls = _post_split_oversample(train_seqs, train_lbls)

    if config.ADVERSARIAL_TRAINING:
        train_seqs, train_lbls = add_adversarial_samples(
            train_seqs, train_lbls,
            noise_std=config.ADVERSARIAL_NOISE_STD,
            ratio=config.ADVERSARIAL_SAMPLE_RATIO,
        )
        print(f"  Augmented train: {tuple(train_seqs.shape)} (val untouched: {len(val_lbls):,})")

    train_ds = SequenceDataset(train_seqs, train_lbls)
    val_ds = SequenceDataset(val_seqs, val_lbls)

    num_workers = getattr(config, 'V2_NUM_WORKERS', 4)
    pin_memory = getattr(config, 'V2_PIN_MEMORY', True) and torch.cuda.is_available()
    persist = num_workers > 0
    prefetch = 4 if num_workers > 0 else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persist,
                              drop_last=True,
                              prefetch_factor=prefetch)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=persist,
                            prefetch_factor=prefetch)

    print(f"  Train samples:   {len(train_ds):,}")
    print(f"  Val samples:     {len(val_ds):,}")

    orig_train_counts = torch.bincount(lbls[train_idx], minlength=config.NUM_CLASSES).float()
    class_weights = torch.zeros(config.NUM_CLASSES)
    active_mask = orig_train_counts > 0
    if active_mask.any():
        inv_freq = 1.0 / orig_train_counts[active_mask].sqrt()
        inv_freq = inv_freq / inv_freq.sum() * active_mask.sum().float()
        class_weights[active_mask] = inv_freq
    class_weights = class_weights.to(device)
    active_classes = active_mask.sum().item()
    print(f"  Active classes:  {active_classes}/{config.NUM_CLASSES}")
    print(f"  Class weights:   {[f'{w:.3f}' for w in class_weights.tolist()]} (from pre-oversample counts)")

    model = get_model(model_variant, num_features=seqs.shape[2])
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model params:    {param_count:,}")

    compiled = False
    if hasattr(torch, 'compile'):
        import platform
        backends_to_try = ["inductor", "aot_eager"] if platform.system() != "Windows" else ["aot_eager"]
        for backend in backends_to_try:
            try:
                model = torch.compile(model, backend=backend)
                compiled = True
                print(f"  torch.compile:   Enabled (backend={backend})")
                break
            except Exception as e:
                print(f"  torch.compile:   {backend} failed ({e}), trying next...")
    if not compiled:
        print(f"  torch.compile:   Disabled (no compatible backend)")

    print(f"{'=' * 70}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    warmup_epochs = config.V2_WARMUP_EPOCHS if is_v2_model else 3
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.V2_COSINE_T0 if is_v2_model else 10,
        T_mult=config.V2_COSINE_TMULT if is_v2_model else 2,
        eta_min=1e-7,
    )
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=config.V2_SWA_LR if is_v2_model else 5e-4)

    ema = EMAModel(model, decay=0.999)

    orig_focal = config.USE_FOCAL_LOSS
    orig_smooth = config.LABEL_SMOOTHING
    config.USE_FOCAL_LOSS = use_focal
    config.LABEL_SMOOTHING = label_smooth
    criterion = get_loss_function(config, class_weights=class_weights)
    config.USE_FOCAL_LOSS = orig_focal
    config.LABEL_SMOOTHING = orig_smooth

    scaler = GradScaler(enabled=use_amp)

    if is_v2_model:
        save_path = config.V2_MODEL_SAVE_PATH
        swa_save_path = config.V2_SWA_MODEL_PATH
    else:
        save_path = config.MODELS_DIR / f"best_{model_variant}.pth"
        swa_save_path = config.MODELS_DIR / f"best_{model_variant}_swa.pth"

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>10}  {'Phase':>6}  {'Time':>5}")
    print(f"  {'-' * 76}")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()
        in_swa = epoch >= swa_start

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        clean_correct = 0
        clean_total = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if model.training:
                B, T, F = x.shape
                n_time_mask = np.random.randint(1, 4)
                for _ in range(n_time_mask):
                    t_idx = np.random.randint(0, T)
                    x[:, t_idx, :] = 0.0
                n_feat_mask = np.random.randint(1, 4)
                for _ in range(n_feat_mask):
                    f_idx = np.random.randint(0, F)
                    x[:, :, f_idx] = 0.0

            web_inj_id = config.meta_label_to_id("Web/Injection")
            has_rare = (y == web_inj_id).any()
            aug_rand = np.random.random()
            use_aug = aug_rand < 0.5 and not has_rare
            if use_aug:
                mixup_alpha = 0.4
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                shuffle_idx = torch.randperm(x.size(0), device=x.device)
                y_a, y_b = y, y[shuffle_idx]

                if np.random.random() < 0.5:
                    x_input = lam * x + (1.0 - lam) * x[shuffle_idx]
                else:
                    T = x.size(1)
                    cut_len = int(T * (1.0 - lam))
                    cut_start = np.random.randint(0, max(T - cut_len, 1))
                    x_input = x.clone()
                    x_input[:, cut_start:cut_start + cut_len, :] = x[shuffle_idx, cut_start:cut_start + cut_len, :]
                    lam = 1.0 - cut_len / T
            else:
                x_input = x
                lam = 1.0

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                out = model(x_input)
                if use_aug:
                    loss = lam * criterion(out, y_a) + (1.0 - lam) * criterion(out, y_b)
                else:
                    loss = criterion(out, y)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            running_loss += loss.item() * x.size(0)
            preds = out.argmax(1)

            if use_aug:
                correct += (lam * (preds == y_a).float() + (1.0 - lam) * (preds == y_b).float()).sum().item()
            else:
                batch_correct = (preds == y).sum().item()
                correct += batch_correct
                clean_correct += batch_correct
                clean_total += x.size(0)

            total += x.size(0)

        train_loss = running_loss / total
        train_acc = clean_correct / max(clean_total, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast(device_type="cuda", enabled=use_amp):
                    out = model(x)
                    loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.perf_counter() - epoch_start

        phase = "SWA" if in_swa else "Train"
        print(f"  {epoch:5d}  {train_loss:10.4f}  {train_acc:8.4f}  "
              f"{val_loss:8.4f}  {val_acc:7.4f}  {current_lr:10.6f}  {phase:>6}  {epoch_time:5.1f}s",
              flush=True)

        if in_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        if not in_swa:
            improved = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
            if improved:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "model_variant": model_variant,
                    "num_features": seqs.shape[2],
                    "num_classes": config.NUM_CLASSES,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "zscore_mean": zscore_mean.squeeze(0),
                    "zscore_std": zscore_std.squeeze(0),
                }
                torch.save(checkpoint, save_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    if epoch >= swa_start:
        bn_ds = SequenceDataset(seqs[train_idx], lbls[train_idx])
        bn_loader = DataLoader(bn_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"\n  Updating SWA batch normalization (original distribution: {len(bn_ds):,} windows)...")
        update_bn(bn_loader, swa_model, device=device)
        swa_checkpoint = {
            "model_state_dict": swa_model.module.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "model_variant": model_variant,
            "num_features": seqs.shape[2],
            "num_classes": config.NUM_CLASSES,
            "epoch": epoch,
            "swa_epochs": epoch - swa_start + 1,
            "zscore_mean": zscore_mean.squeeze(0),
            "zscore_std": zscore_std.squeeze(0),
        }
        torch.save(swa_checkpoint, swa_save_path)
        print(f"  SWA model saved to: {swa_save_path}")

    ema_save_path = save_path.parent / f"{save_path.stem}_ema.pth"
    ema_checkpoint = {
        "model_state_dict": ema.state_dict(),
        "model_variant": model_variant,
        "num_features": seqs.shape[2],
        "num_classes": config.NUM_CLASSES,
        "epoch": epoch,
        "zscore_mean": zscore_mean.squeeze(0),
        "zscore_std": zscore_std.squeeze(0),
    }
    torch.save(ema_checkpoint, ema_save_path)
    print(f"  EMA model saved to: {ema_save_path}")

    print(f"\n{'=' * 70}")
    print("  TRAINING COMPLETE")
    print(f"  Best val accuracy: {best_val_acc:.4f} (loss: {best_val_loss:.4f})")
    print(f"  Model saved to: {save_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IPS model (unified pipeline)")
    parser.add_argument("--model", type=str, default=None,
                        choices=ALL_VARIANTS,
                        help="Model variant (default: config.MODEL_VARIANT)")
    parser.add_argument("--focal", action="store_true", default=None, help="Use Focal Loss (default: from config)")
    parser.add_argument("--no-focal", action="store_true", help="Disable Focal Loss")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--all", action="store_true", help="Train ALL model variants")
    args = parser.parse_args()

    if args.no_focal:
        focal_setting = False
    elif args.focal:
        focal_setting = True
    else:
        focal_setting = None

    if args.epochs:
        config.NUM_EPOCHS = args.epochs
        config.V2_NUM_EPOCHS = args.epochs

    if args.all:
        results = {}
        for variant in ALL_VARIANTS:
            print(f"\n{'#' * 70}")
            print(f"  TRAINING: {variant}")
            print(f"{'#' * 70}\n")
            try:
                train(model_variant=variant, use_focal=focal_setting)
                results[variant] = "SUCCESS"
            except Exception as e:
                print(f"ERROR training {variant}: {e}")
                results[variant] = f"FAILED: {e}"
                import traceback
                traceback.print_exc()
        print(f"\n{'=' * 70}")
        print(f"  ALL TRAINING COMPLETE")
        print(f"{'=' * 70}")
        for v, r in results.items():
            print(f"  {v:25s} -> {r}")
    else:
        train(model_variant=args.model, use_focal=focal_setting)

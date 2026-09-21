import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src import clean_state_dict


ENSEMBLE_SEEDS = [42, 123, 256, 789, 1337]


def train_ensemble(epochs=100, seeds=None):
    from src.train import train

    seeds = seeds or ENSEMBLE_SEEDS

    print(f"\n{'=' * 60}")
    print(f"  ENSEMBLE TRAINING ({len(seeds)} models)")
    print(f"  Seeds: {seeds}")
    print(f"{'=' * 60}\n")

    for i, seed in enumerate(seeds):
        print(f"\n{'─' * 60}")
        print(f"  Training model {i+1}/{len(seeds)} (seed={seed})")
        print(f"{'─' * 60}")

        save_path = config.MODELS_DIR / f"ensemble_v2_seed{seed}.pth"

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train(model_variant="model_v2")

    print(f"\n{'=' * 60}")
    print(f"  ENSEMBLE TRAINING COMPLETE")
    print(f"{'=' * 60}\n")


def load_ensemble(device=None, seeds=None):
    from src.model_v2 import SentinelV2

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = seeds or ENSEMBLE_SEEDS
    models = []

    for seed in seeds:
        path = config.MODELS_DIR / f"ensemble_v2_seed{seed}.pth"
        if not path.exists():
            print(f"  ✗ Missing: {path}")
            continue

        model = SentinelV2(
            num_features=config.V2_NUM_FEATURES,
            num_classes=config.NUM_CLASSES,
        )
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(clean_state_dict(ckpt.get("model_state_dict", ckpt)))
        model.to(device)
        model.eval()
        models.append(model)

    print(f"  Loaded {len(models)}/{len(seeds)} ensemble models")
    return models


def ensemble_predict(models, x):
    all_probs = []
    with torch.no_grad():
        for model in models:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    preds = avg_probs.argmax(dim=1)
    confidence = avg_probs.max(dim=1).values

    return preds, avg_probs, confidence


def evaluate_ensemble(seeds=None):
    from src.sessionization import load_sequences
    from src.split_utils import block_split
    from sklearn.metrics import (accuracy_score, f1_score,
                                  matthews_corrcoef, classification_report)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seqs, lbls, _ = load_sequences()
    n = len(lbls)
    _, _, test_idx = block_split(n)
    test_seqs = seqs[test_idx].to(device)
    test_lbls = lbls[test_idx].numpy()

    models = load_ensemble(device, seeds)
    if len(models) < 2:
        print("  ERROR: Need at least 2 models for ensemble")
        return

    print(f"\n{'=' * 60}")
    print(f"  ENSEMBLE EVALUATION ({len(models)} models)")
    print(f"{'=' * 60}\n")

    batch_size = 512
    all_preds = []
    for i in range(0, len(test_seqs), batch_size):
        batch = test_seqs[i:i + batch_size]
        preds, _, _ = ensemble_predict(models, batch)
        all_preds.append(preds.cpu().numpy())

    ensemble_preds = np.concatenate(all_preds)

    acc = accuracy_score(test_lbls, ensemble_preds)
    f1 = f1_score(test_lbls, ensemble_preds, average="weighted")
    mcc = matthews_corrcoef(test_lbls, ensemble_preds)

    print(f"  Ensemble Accuracy: {acc:.4f}")
    print(f"  Ensemble F1:       {f1:.4f}")
    print(f"  Ensemble MCC:      {mcc:.4f}")

    class_names = config.get_class_names()
    print(f"\n  Per-class report:")
    print(classification_report(test_lbls, ensemble_preds,
                                 target_names=class_names, digits=4))

    print(f"\n  Individual model accuracies:")
    for i, model in enumerate(models):
        individual_preds = []
        for j in range(0, len(test_seqs), batch_size):
            batch = test_seqs[j:j + batch_size]
            with torch.no_grad():
                out = model(batch)
            individual_preds.append(out.argmax(1).cpu().numpy())
        ind_preds = np.concatenate(individual_preds)
        ind_acc = accuracy_score(test_lbls, ind_preds)
        print(f"    Model {i}: {ind_acc:.4f}")

    print(f"\n  Ensemble improvement: +{(acc - ind_acc)*100:.2f}% over last model")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble training/evaluation")
    parser.add_argument("--mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "train":
        train_ensemble(epochs=args.epochs)
    else:
        evaluate_ensemble()

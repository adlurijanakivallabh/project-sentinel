import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src import clean_state_dict
from src.sessionization import load_sequences
from src.train import get_model
from src.split_utils import block_split


def _load_model_and_data(n_samples=500):
    device = torch.device("cpu")

    checkpoint = torch.load(config.MODEL_CHECKPOINT_PATH, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        variant = checkpoint.get("model_variant", "cnn_lstm")
        num_features = checkpoint.get("num_features", len(config.FEATURE_COLUMNS))
        num_classes = checkpoint.get("num_classes", config.NUM_CLASSES)
        state_dict = checkpoint["model_state_dict"]
    else:
        variant = "cnn_lstm"
        state_dict = checkpoint
        seqs, _, _ = load_sequences()
        num_features = seqs.shape[2]
        num_classes = config.NUM_CLASSES

    model = get_model(variant, num_features=num_features, num_classes=num_classes)
    model.load_state_dict(clean_state_dict(state_dict))
    model.eval()

    seqs, lbls, _ = load_sequences()
    _, _, test_idx = block_split(len(lbls))

    subset_idx = test_idx[:n_samples]
    X = seqs[subset_idx]
    y = lbls[subset_idx].numpy()

    return model, X, y, variant


def visualize_attention(n_examples=5):
    print("\n[explainability] Attention Weight Visualization")
    print("=" * 50)

    model, X, y, variant = _load_model_and_data(n_samples=1000)

    if not hasattr(model, "use_attention") or not model.use_attention:
        print("  Model does not have attention. Skipping.")
        return

    class_names = config.get_class_names()
    num_classes = len(class_names)

    fig, axes = plt.subplots(num_classes, 1, figsize=(12, 3 * num_classes))
    if num_classes == 1:
        axes = [axes]

    for cls_id in range(num_classes):
        mask = y == cls_id
        if mask.sum() == 0:
            continue

        cls_seqs = X[mask][:n_examples]

        all_weights = []
        with torch.no_grad():
            for i in range(len(cls_seqs)):
                seq = cls_seqs[i:i + 1]
                _, attn_w = model(seq, return_attention=True)
                all_weights.append(attn_w.squeeze().numpy())

        if not all_weights:
            continue

        avg_weights = np.mean(all_weights, axis=0)

        ax = axes[cls_id]
        ax.bar(range(len(avg_weights)), avg_weights, color="#3498db", alpha=0.8)
        ax.set_title(f"{class_names[cls_id]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("CNN Time Step")
        ax.set_ylabel("Attention Weight")
        ax.set_ylim(0, max(avg_weights) * 1.3)

    plt.suptitle("Attention Weights by Attack Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = config.EXPLAINABILITY_DIR / "attention_weights.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved to {path}")


def feature_importance_rf():
    print("\n[explainability] Random Forest Feature Importance")
    print("=" * 50)

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("  sklearn not available. Skipping.")
        return

    _, X, y, _ = _load_model_and_data(n_samples=2000)

    X_flat = X.numpy().mean(axis=1)

    feature_names = list(config.FEATURE_COLUMNS)
    if config.TEMPORAL_FEATURES_ENABLED:
        feature_names += config.TEMPORAL_FEATURE_NAMES

    feature_names = feature_names[:X_flat.shape[1]]

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_flat, y)

    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\n  Feature Importance Ranking:")
    print(f"  {'Rank':>4}  {'Feature':>30}  {'Importance':>10}")
    print(f"  {'-' * 48}")
    for rank, idx in enumerate(sorted_idx, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        print(f"  {rank:4d}  {name:>30}  {importances[idx]:10.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(feature_names))
    top_idx = sorted_idx[:top_n][::-1]

    names = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in top_idx]
    values = importances[top_idx]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax.barh(range(top_n), values, color=colors, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Top Feature Importance (Random Forest)", fontsize=14)
    plt.tight_layout()
    path = config.EXPLAINABILITY_DIR / "feature_importance_rf.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved to {path}")

    threshold = 0.01
    drop_features = [feature_names[i] for i in sorted_idx if importances[i] < threshold
                     and i < len(feature_names)]
    if drop_features:
        print(f"\n  [!] Features with importance < {threshold} (consider dropping):")
        for f in drop_features:
            print(f"    - {f}")


def shap_analysis():
    print("\n[explainability] SHAP Feature Importance")
    print("=" * 50)

    try:
        import shap
    except ImportError:
        print("  SHAP not installed. Install with: pip install shap")
        print("  Skipping SHAP analysis.")
        return

    _, X, y, _ = _load_model_and_data(n_samples=500)

    X_flat = X.numpy().mean(axis=1)

    feature_names = list(config.FEATURE_COLUMNS)
    if config.TEMPORAL_FEATURES_ENABLED:
        feature_names += config.TEMPORAL_FEATURE_NAMES
    feature_names = feature_names[:X_flat.shape[1]]

    from sklearn.ensemble import RandomForestClassifier
    model_rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model_rf.fit(X_flat[:400], y[:400])

    explainer = shap.TreeExplainer(model_rf)
    shap_values = explainer.shap_values(X_flat[:100])

    fig, ax = plt.subplots(figsize=(10, 8))
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    top_n = min(20, len(feature_names))
    top_idx = sorted_idx[:top_n][::-1]

    names = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in top_idx]
    values = mean_abs_shap[top_idx]

    ax.barh(range(top_n), values, color="#e74c3c", alpha=0.8, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("SHAP Global Feature Importance", fontsize=14)
    plt.tight_layout()
    path = config.EXPLAINABILITY_DIR / "shap_global_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Global importance plot saved to {path}")

    class_names = config.get_class_names()
    if isinstance(shap_values, list):
        shap_list = shap_values
    elif shap_values.ndim == 3:
        shap_list = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
    else:
        shap_list = [shap_values]

    for cls_id, sv in enumerate(shap_list):
        if cls_id >= len(class_names):
            break
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                sv, X_flat[:100],
                feature_names=feature_names,
                show=False,
                max_display=15,
            )
            plt.title(f"SHAP Summary — {class_names[cls_id]}", fontsize=14)
            plt.tight_layout()
            path = config.EXPLAINABILITY_DIR / f"shap_class_{cls_id}_{class_names[cls_id].replace('/', '_')}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Class {class_names[cls_id]} plot saved to {path}")
        except Exception as e:
            print(f"  Class {class_names[cls_id]} plot failed: {e}")

    print("  SHAP analysis complete.")


def run_all():
    feature_importance_rf()
    visualize_attention()
    shap_analysis()


if __name__ == "__main__":
    run_all()

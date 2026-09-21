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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqs, lbls, _ = load_sequences()
    _, _, test_idx = block_split(len(lbls))
    test_seqs, test_lbls = seqs[test_idx], lbls[test_idx]

    ckpt = torch.load(config.MODEL_CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = get_model(ckpt.get("model_variant", "cnn_lstm_attention"), num_features=ckpt.get("num_features", 30))
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    model.to(device).eval()

    names = config.get_class_names()

    class_attention = {}
    for cls_id in range(6):
        mask = test_lbls == cls_id
        cls_seqs = test_seqs[mask][:500].to(device)
        if len(cls_seqs) == 0:
            continue
        with torch.no_grad():
            _, attn_weights = model(cls_seqs, return_attention=True)
        class_attention[cls_id] = attn_weights.cpu().numpy()
        print(f"{names[cls_id]}: attn shape={attn_weights.shape}, n={len(cls_seqs)}")

    cnn_steps = class_attention[0].shape[1]
    print(f"CNN output timesteps: {cnn_steps}")

    flows_per_step = 20 // cnn_steps
    step_labels = []
    for i in range(cnn_steps):
        start = i * flows_per_step
        end = start + flows_per_step - 1
        step_labels.append(f"Flows {start}-{end}")

    colors_list = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "CNN-LSTM-Attention: Per-Class Temporal Focus\n"
        "(Which flow groups the model attends to for each attack type)",
        fontsize=14, fontweight="bold",
    )

    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        attn = class_attention[idx]
        mean_a = np.mean(attn, axis=0)
        std_a = np.std(attn, axis=0)

        bars = ax.bar(
            range(cnn_steps), mean_a, yerr=std_a, capsize=3,
            color=colors_list[idx], alpha=0.8, edgecolor="black", linewidth=0.5,
        )

        max_idx = np.argmax(mean_a)
        bars[max_idx].set_color("red")
        bars[max_idx].set_alpha(0.9)

        ax.set_title(f"{names[idx]}", fontsize=12, fontweight="bold")
        ax.set_xticks(range(cnn_steps))
        ax.set_xticklabels(step_labels, fontsize=8, rotation=15)
        ax.set_ylabel("Attention Weight")
        ax.set_ylim(0, max(mean_a + std_a) * 1.3)

    plt.tight_layout()
    path1 = config.FIGURES_DIR / "attention_per_class.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path1}")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(cnn_steps)
    width = 0.13

    for idx in range(6):
        mean_a = np.mean(class_attention[idx], axis=0)
        offset = (idx - 2.5) * width
        ax.bar(
            x + offset, mean_a, width, label=names[idx],
            color=colors_list[idx], alpha=0.85, edgecolor="black", linewidth=0.3,
        )

    ax.set_title("Attention Weight Comparison Across All Classes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temporal Region (flow groups)", fontsize=12)
    ax.set_ylabel("Average Attention Weight", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    path2 = config.FIGURES_DIR / "attention_comparison.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path2}")

    print()
    print("=== ATTENTION FOCUS SUMMARY ===")
    for cls_id in range(6):
        mean_a = np.mean(class_attention[cls_id], axis=0)
        max_step = np.argmax(mean_a)
        early_pct = mean_a[0] / mean_a.sum() * 100
        late_pct = mean_a[-1] / mean_a.sum() * 100
        weights_str = ", ".join(f"{v:.3f}" for v in mean_a)
        print(
            f"{names[cls_id]:25s}: Peak={step_labels[max_step]}, "
            f"Early={early_pct:.1f}%, Late={late_pct:.1f}%, "
            f"Weights=[{weights_str}]"
        )


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.train import get_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


UNSW_LABEL_MAP = {
    "Normal":          0,
    "DoS":             1,
    "Generic":         2,
    "Reconnaissance":  3,
    "Analysis":        3,
    "Fuzzers":         4,
    "Exploits":        4,
    "Shellcode":       5,
    "Backdoor":        5,
    "Worms":           5,
}

UNSW_FEATURE_MAP = {
    "dur":                "Flow Duration",
    "spkts":              "Total Fwd Packets",
    "dpkts":              "Total Backward Packets",
    "sbytes":             "Total Length of Fwd Packets",
    "dbytes":             "Total Length of Bwd Packets",
    "rate":               "Flow Packets/s",
    "sload":              "Flow Bytes/s",
    "sinpkt":             "Flow IAT Mean",
    "dinpkt":             "Flow IAT Std",
    "sjit":               "Flow IAT Max",
    "djit":               "Flow IAT Min",
    "smean":              "Fwd Packet Length Mean",
    "dmean":              "Bwd Packet Length Mean",
    "sttl":               "Fwd Packet Length Max",
    "dttl":               "Bwd Packet Length Max",
    "sloss":              "Fwd Packet Length Min",
    "dloss":              "Bwd Packet Length Min",
    "swin":               "Fwd Packet Length Std",
    "dwin":               "Bwd Packet Length Std",
    "tcprtt":             "Fwd PSH Flags",
    "synack":             "Bwd PSH Flags",
    "ackdat":             "Fwd URG Flags",
    "trans_depth":        "Bwd URG Flags",
    "ct_srv_src":         "Destination Port",
    "ct_dst_ltm":         "Protocol",
}


def load_unsw_nb15():
    data_dir = project_root / "data" / "unsw-nb15"
    test_file = data_dir / "UNSW_NB15_testing-set.csv"

    print(f"[cross-eval] Loading {test_file.name}...")
    df = pd.read_csv(test_file)
    print(f"[cross-eval] Raw samples: {len(df):,}")

    df["attack_cat"] = df["attack_cat"].str.strip()
    df["meta_class"] = df["attack_cat"].map(UNSW_LABEL_MAP)
    unknown = df["meta_class"].isna().sum()
    if unknown > 0:
        print(f"[cross-eval] WARNING: {unknown} unmapped labels, dropping them")
        df = df.dropna(subset=["meta_class"])
    df["meta_class"] = df["meta_class"].astype(int)

    print("[cross-eval] Label distribution:")
    class_names = config.get_class_names()
    for cls_id in sorted(df["meta_class"].unique()):
        name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        count = (df["meta_class"] == cls_id).sum()
        print(f"  {name:30s}: {count:,}")

    features = np.zeros((len(df), len(config.FEATURE_COLUMNS)), dtype=np.float32)
    mapped_count = 0
    for unsw_col, our_col in UNSW_FEATURE_MAP.items():
        if unsw_col in df.columns and our_col in config.FEATURE_COLUMNS:
            feat_idx = config.FEATURE_COLUMNS.index(our_col)
            features[:, feat_idx] = df[unsw_col].values.astype(np.float32)
            mapped_count += 1

    print(f"[cross-eval] Mapped {mapped_count}/{len(config.FEATURE_COLUMNS)} features "
          f"({len(config.FEATURE_COLUMNS) - mapped_count} zeros)")

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    labels = df["meta_class"].values
    return features, labels


def create_windows(features, labels, window_size=20):
    n_windows = len(features) // window_size
    X = features[:n_windows * window_size].reshape(n_windows, window_size, -1)

    y_per_flow = labels[:n_windows * window_size].reshape(n_windows, window_size)
    y = np.array([np.bincount(row, minlength=config.NUM_CLASSES).argmax() for row in y_per_flow])

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def run_cross_dataset_eval():
    print("=" * 60)
    print("  FIX #3: CROSS-DATASET GENERALIZATION TEST")
    print("  Model trained on: CIC-IDS2017, CSE-CIC-2018, CICIoT2023,")
    print("                    UNSW-NB15, CIC-DDoS2019")
    print("  Testing on:       UNSW-NB15 TESTING SET (held-out)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, labels = load_unsw_nb15()

    X, y = create_windows(features, labels, window_size=config.SEQUENCE_LENGTH)
    print(f"\n[cross-eval] Windows: {len(X):,} × {config.SEQUENCE_LENGTH} × {X.shape[2]}")

    print("[cross-eval] Loading trained model...")
    ckpt = torch.load(config.MODEL_CHECKPOINT_PATH, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        variant = ckpt.get("model_variant", config.MODEL_VARIANT)
        num_features = ckpt.get("num_features", X.shape[2])
        state_dict = ckpt["model_state_dict"]
    else:
        variant = config.MODEL_VARIANT
        num_features = 30
        state_dict = ckpt

    if X.shape[2] < num_features:
        pad_size = num_features - X.shape[2]
        print("[cross-eval] Padding %d -> %d features (%d temporal zeros)" % (X.shape[2], num_features, pad_size))
        padding = torch.zeros(X.shape[0], X.shape[1], pad_size)
        X = torch.cat([X, padding], dim=2)

    model = get_model(variant, num_features=num_features)
    from src import clean_state_dict
    model.load_state_dict(clean_state_dict(state_dict))
    model.to(device)
    model.eval()

    print("[cross-eval] Running inference...")
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            batch = X[i:i+256].to(device)
            out = model(batch)
            preds.extend(out.argmax(1).cpu().numpy())
    preds = np.array(preds)
    y_np = y.numpy()

    accuracy = (preds == y_np).mean()
    class_names = config.get_class_names()

    print("\n" + "=" * 60)
    print(f"  CROSS-DATASET ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(y_np, preds, target_names=class_names, digits=4,
                                zero_division=0))

    cm = confusion_matrix(y_np, preds, labels=list(range(len(class_names))))
    print("Confusion Matrix:")
    header = "{:>25}".format("TRUE \\ PRED") + "".join(["{:>10}".format(n[:9]) for n in class_names])
    print(header)
    print("-" * (25 + 10 * len(class_names)))
    for i, name in enumerate(class_names):
        row = "{:>25}".format(name) + "".join(["{:>10}".format(cm[i][j]) for j in range(len(class_names))])
        print(row)

    results_path = config.RESULTS_DIR / "cross_dataset_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Cross-Dataset Generalization Test\n")
        f.write(f"{'='*60}\n")
        f.write(f"Tested on: UNSW-NB15 testing set (held-out)\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(y_np, preds, target_names=class_names, digits=4,
                                      zero_division=0))
    print(f"\n[cross-eval] Results saved to {results_path}")

    return accuracy


if __name__ == "__main__":
    run_cross_dataset_eval()

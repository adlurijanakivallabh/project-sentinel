import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config


def _compute_temporal_features(window: np.ndarray, feature_cols: list) -> np.ndarray:
    seq_len = window.shape[0]
    temporal = np.zeros((seq_len, 5), dtype=np.float32)

    def _idx(name):
        try:
            return feature_cols.index(name)
        except ValueError:
            return None

    bytes_idx = _idx("Flow Bytes/s")
    pkts_idx = _idx("Flow Packets/s")
    fwd_pkts_idx = _idx("Total Fwd Packets")
    bwd_pkts_idx = _idx("Total Backward Packets")
    duration_idx = _idx("Flow Duration")

    if bytes_idx is not None:
        vals = window[:, bytes_idx]
        temporal[1:, 0] = np.diff(vals)

    if pkts_idx is not None:
        vals = window[:, pkts_idx]
        temporal[1:, 1] = np.diff(vals)

    if fwd_pkts_idx is not None:
        vals = window[:, fwd_pkts_idx]
        for i in range(seq_len):
            start = max(0, i - 4)
            temporal[i, 2] = np.mean(vals[start:i + 1])

    if bwd_pkts_idx is not None:
        vals = window[:, bwd_pkts_idx]
        for i in range(seq_len):
            start = max(0, i - 4)
            temporal[i, 3] = np.mean(vals[start:i + 1])

    if duration_idx is not None:
        var_val = np.var(window[:, duration_idx])
        temporal[:, 4] = var_val

    return temporal


def build_sequences_from_dataframe(df):
    seq_len = config.SEQUENCE_LENGTH or 20
    step = config.SEQUENCE_STEP or seq_len

    feature_cols = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    if not feature_cols:
        feature_cols = [c for c in df.columns
                        if c not in ("Label", "class_id", "meta_label", "flow_id")]

    feats = df[feature_cols].values.astype("float32")

    if "class_id" not in df.columns:
        if "meta_label" in df.columns:
            df["class_id"] = df["meta_label"].apply(config.meta_label_to_id)
        else:
            label_map = config.get_label_map()
            df["meta_label"] = df["Label"].map(
                lambda x: label_map.get(str(x).strip(), "Normal") if isinstance(x, str) else "Normal"
            )
            df["class_id"] = df["meta_label"].map(config.meta_label_to_id)

    clabels = df["class_id"].values.astype("int64")

    sequences = []
    labels = []

    n = len(df)
    for start in range(0, n - seq_len + 1, step):
        end = start + seq_len
        window_feats = feats[start:end]
        window_lbls = clabels[start:end]

        normal_id = config.meta_label_to_id("Normal")
        uniq, counts = np.unique(window_lbls, return_counts=True)

        is_attack = uniq != normal_id
        if is_attack.any():
            attack_uniq = uniq[is_attack]
            attack_counts = counts[is_attack]
            majority = int(attack_uniq[np.argmax(attack_counts)])
        else:
            majority = int(normal_id)

        if config.TEMPORAL_FEATURES_ENABLED:
            temporal = _compute_temporal_features(window_feats, feature_cols)
            window_feats = np.concatenate([window_feats, temporal], axis=1)

        sequences.append(window_feats)
        labels.append(majority)

    if not sequences:
        num_feat = len(feature_cols) + (5 if config.TEMPORAL_FEATURES_ENABLED else 0)
        return torch.tensor([]).reshape(0, seq_len, num_feat), torch.tensor([]), []

    return torch.tensor(np.stack(sequences)), torch.tensor(labels), []


def save_sequences(seqs, lbls, path):
    torch.save({"sequences": seqs, "labels": lbls}, path)
    print(f"[sessionization] Saved {len(lbls):,} sequences "
          f"(shape {tuple(seqs.shape)}) to {path}")


def load_sequences(path=None):
    if path is None:
        path = config.SEQUENCES_PATH
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        return data["sequences"], data["labels"], data.get("meta", [])
    return data


if __name__ == "__main__":
    csv_path = config.DATASETS_DIR / "combined_preprocessed.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} missing. Run data_preprocessing first.")
        sys.exit(1)

    print("Loading preprocessed CSV...")
    df = pd.read_csv(csv_path)
    print(f"Building sequences from {len(df):,} rows...")
    print(f"Temporal features: {'ENABLED' if config.TEMPORAL_FEATURES_ENABLED else 'DISABLED'}")

    seqs, lbls, _ = build_sequences_from_dataframe(df)
    save_sequences(seqs, lbls, config.SEQUENCES_PATH)

    print(f"\nSequence shape: {tuple(seqs.shape)}")
    print(f"  Base features: {len(config.FEATURE_COLUMNS)}")
    if config.TEMPORAL_FEATURES_ENABLED:
        print(f"  Temporal features: {len(config.TEMPORAL_FEATURE_NAMES)}")
        print(f"  Total features per flow: {seqs.shape[2]}")

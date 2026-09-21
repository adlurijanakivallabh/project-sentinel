import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.sessionization import load_sequences
from src.split_utils import block_split


def compute_feature_importance():
    from sklearn.feature_selection import mutual_info_classif, f_classif

    print("Loading sequences...")
    seqs, lbls, _ = load_sequences()
    n = len(lbls)
    print(f"  Shape: {tuple(seqs.shape)}, Classes: {config.NUM_CLASSES}")

    train_idx, _, _ = block_split(n)
    train_seqs = seqs[train_idx]
    train_lbls = lbls[train_idx]
    print(f"  Train partition: {len(train_idx):,} sequences")

    train_flat = train_seqs.reshape(-1, train_seqs.shape[-1])
    zscore_mean = train_flat.mean(dim=0, keepdim=True)
    zscore_std = train_flat.std(dim=0, keepdim=True).clamp(min=1e-8)
    train_seqs = ((train_seqs - zscore_mean) / zscore_std).clamp(-10, 10)

    X = train_seqs.mean(dim=1).numpy()
    y = train_lbls.numpy()
    print(f"  Feature matrix: {X.shape}")

    base_features = list(config.FEATURE_COLUMNS)
    temporal_features = list(config.TEMPORAL_FEATURE_NAMES) if config.TEMPORAL_FEATURES_ENABLED else []
    all_features = base_features + temporal_features
    
    if len(all_features) > X.shape[1]:
        all_features = all_features[:X.shape[1]]
    elif len(all_features) < X.shape[1]:
        all_features += [f"Feature_{i}" for i in range(len(all_features), X.shape[1])]

    print("\nComputing Mutual Information scores...")
    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42, n_neighbors=5)

    print("Computing ANOVA F-statistic scores...")
    f_scores, p_values = f_classif(X, y)

    results = pd.DataFrame({
        "Feature": all_features,
        "MI_Score": mi_scores,
        "MI_Rank": 0,
        "ANOVA_F": f_scores,
        "ANOVA_p": p_values,
        "ANOVA_Rank": 0,
    })

    results["ANOVA_F"] = results["ANOVA_F"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    results["ANOVA_p"] = results["ANOVA_p"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    results["MI_Score"] = results["MI_Score"].fillna(0.0)

    constant_mask = results["ANOVA_F"] == 0.0
    if constant_mask.any():
        const_names = results.loc[constant_mask, "Feature"].tolist()
        print(f"\n  WARNING: Constant features detected (zero variance): {const_names}")

    results["MI_Rank"] = results["MI_Score"].rank(ascending=False, na_option='bottom').astype(int)
    results["ANOVA_Rank"] = results["ANOVA_F"].rank(ascending=False, na_option='bottom').astype(int)
    results["Avg_Rank"] = ((results["MI_Rank"] + results["ANOVA_Rank"]) / 2)
    results = results.sort_values("Avg_Rank")

    print("\n" + "=" * 90)
    print("  FEATURE IMPORTANCE RANKING (MI + ANOVA)")
    print("=" * 90)
    print(f"  {'Rank':>4}  {'Feature':<30}  {'MI Score':>10}  {'MI Rank':>8}  {'ANOVA F':>10}  {'ANOVA Rank':>10}")
    print(f"  {'-'*84}")
    
    for _, row in results.iterrows():
        avg_rank = row['Avg_Rank']
        marker = " *" if avg_rank <= 10 else " ." if avg_rank <= 20 else "  "
        print(f"  {avg_rank:4.1f}  {row['Feature']:<30}  {row['MI_Score']:10.4f}  {int(row['MI_Rank']):8d}  "
              f"{row['ANOVA_F']:10.1f}  {int(row['ANOVA_Rank']):10d}{marker}")

    print(f"\n  Top 10 features (by average MI + ANOVA rank):")
    top10 = results.head(10)
    for _, row in top10.iterrows():
        print(f"    {row['Avg_Rank']:4.1f}  {row['Feature']}")

    bottom5 = results.tail(5)
    print(f"\n  Bottom 5 features (candidates for pruning):")
    for _, row in bottom5.iterrows():
        print(f"    {row['Avg_Rank']:4.1f}  {row['Feature']}")

    out_path = config.RESULTS_DIR / "feature_importance.csv"
    results.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

    return results


if __name__ == "__main__":
    compute_feature_importance()

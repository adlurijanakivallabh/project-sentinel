import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

NON_FEATURE_COLS = {'Label', 'meta_label', 'class_id'}


def load_data(sample_size=200000):
    csv_path = project_root / "results" / "datasets" / "cicids2017_preprocessed.csv"
    print(f"[1/5] Loading data from {csv_path.name}...")
    
    df_head = pd.read_csv(csv_path, nrows=5)
    features = [c for c in df_head.columns if c not in NON_FEATURE_COLS]
    print(f"       Detected {len(features)} features")
    
    total_rows = sum(1 for _ in open(csv_path)) - 1
    print(f"       Total rows: {total_rows:,}")
    
    if total_rows > sample_size:
        skip_ratio = 1 - (sample_size / total_rows)
        np.random.seed(42)
        df = pd.read_csv(csv_path, skiprows=lambda i: i > 0 and np.random.random() < skip_ratio)
    else:
        df = pd.read_csv(csv_path)
    
    print(f"       Sampled: {len(df):,} rows")
    return df, features


def compute_correlation_matrix(df, features):
    print(f"\n[2/5] Computing Pearson correlation matrix ({len(features)}×{len(features)})...")
    
    feature_df = df[features].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    corr_matrix = feature_df.corr(method='pearson')
    return corr_matrix


def find_redundant_pairs(corr_matrix, threshold=0.95):
    print(f"\n[3/5] Finding redundant pairs (|r| > {threshold})...")
    
    redundant = []
    features = corr_matrix.columns.tolist()
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = abs(corr_matrix.iloc[i, j])
            if r > threshold:
                redundant.append({
                    'Feature_A': features[i],
                    'Feature_B': features[j],
                    'Correlation': corr_matrix.iloc[i, j],
                    'Abs_Correlation': r
                })
    
    redundant.sort(key=lambda x: x['Abs_Correlation'], reverse=True)
    
    if redundant:
        print(f"       Found {len(redundant)} redundant pairs:")
        for p in redundant:
            print(f"       • {p['Feature_A']} ↔ {p['Feature_B']}: r={p['Correlation']:.4f}")
    else:
        print(f"       No pairs found with |r| > {threshold}")
    
    return redundant


def compute_feature_importance(df, features):
    print(f"\n[4/5] Computing RF feature importance...")
    
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = df['class_id'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    acc = rf.score(X_test, y_test)
    print(f"       RF test accuracy: {acc:.4f}")
    
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n       Feature Importance Ranking:")
    for _, row in importances.iterrows():
        bar = '█' * int(row['Importance'] * 200)
        print(f"       {row['Feature']:35s} {row['Importance']:.4f}  {bar}")
    
    return importances


def recommend_pruning(redundant_pairs, importances, corr_threshold=0.95):
    print(f"\n[5/5] Generating pruning recommendations...")
    
    importance_dict = dict(zip(importances['Feature'], importances['Importance']))
    
    features_to_drop = set()
    drop_reasons = {}
    
    for pair in redundant_pairs:
        fa, fb = pair['Feature_A'], pair['Feature_B']
        imp_a = importance_dict.get(fa, 0)
        imp_b = importance_dict.get(fb, 0)
        
        if imp_a >= imp_b:
            drop = fb
            keep = fa
        else:
            drop = fa
            keep = fb
        
        if drop not in features_to_drop:
            features_to_drop.add(drop)
            drop_reasons[drop] = f"Redundant with {keep} (r={pair['Correlation']:.3f}), lower importance ({importance_dict[drop]:.4f} vs {importance_dict[keep]:.4f})"
    
    max_imp = importances['Importance'].max()
    for _, row in importances.iterrows():
        if row['Importance'] < max_imp * 0.005 and row['Feature'] not in features_to_drop:
            features_to_drop.add(row['Feature'])
            drop_reasons[row['Feature']] = f"Near-zero importance ({row['Importance']:.6f})"
    
    kept_features = [f for f in importances['Feature'] if f not in features_to_drop]
    
    print(f"\n       ═══════════════════════════════════════════")
    print(f"       PRUNING RECOMMENDATION")
    print(f"       ═══════════════════════════════════════════")
    print(f"       Original features: {len(importances)}")
    print(f"       Recommended drops: {len(features_to_drop)}")
    print(f"       Final feature count: {len(kept_features)}")
    print(f"\n       Features to DROP:")
    for f in sorted(features_to_drop):
        print(f"       ✗ {f}")
        print(f"         Reason: {drop_reasons[f]}")
    print(f"\n       Features to KEEP ({len(kept_features)}):")
    for f in kept_features:
        print(f"       ✓ {f}")
    
    return kept_features, features_to_drop, drop_reasons


def plot_correlation_heatmap(corr_matrix, output_path):
    fig, ax = plt.subplots(figsize=(16, 14))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'},
        annot_kws={'size': 6},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Matrix (25 CICFlowMeter Features)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"       Saved: {output_path}")


def plot_feature_importance(importances, output_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importances)))
    
    bars = ax.barh(
        range(len(importances)),
        importances['Importance'].values,
        color=colors,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances['Feature'].values, fontsize=9)
    ax.set_xlabel('Feature Importance (Random Forest)', fontsize=12)
    ax.set_title('Feature Importance Ranking — RF (100 trees, balanced)', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"       Saved: {output_path}")


def save_report(importances, redundant_pairs, kept, dropped, reasons, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("FEATURE CORRELATION ANALYSIS & PRUNING REPORT\n")
        f.write("Project Sentinel — MTech Thesis\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. FEATURE IMPORTANCE RANKING (Random Forest)\n")
        f.write("-" * 50 + "\n")
        for _, row in importances.iterrows():
            f.write(f"  {row['Feature']:35s}  {row['Importance']:.6f}\n")
        
        f.write(f"\n2. REDUNDANT PAIRS (|r| > 0.95)\n")
        f.write("-" * 50 + "\n")
        if redundant_pairs:
            for p in redundant_pairs:
                f.write(f"  {p['Feature_A']} ↔ {p['Feature_B']}: r={p['Correlation']:.4f}\n")
        else:
            f.write("  None found.\n")
        
        f.write(f"\n3. PRUNING RECOMMENDATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Original: {len(importances)} features\n")
        f.write(f"  Drop:     {len(dropped)} features\n")
        f.write(f"  Keep:     {len(kept)} features\n\n")
        
        f.write("  Features to DROP:\n")
        for feat in sorted(dropped):
            f.write(f"    ✗ {feat}\n")
            f.write(f"      {reasons[feat]}\n")
        
        f.write(f"\n  Features to KEEP ({len(kept)}):\n")
        for feat in kept:
            f.write(f"    ✓ {feat}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"       Saved: {output_path}")


def main():
    print("=" * 60)
    print("  FEATURE CORRELATION ANALYSIS & PRUNING")
    print("  Project Sentinel — Phase 1a")
    print("=" * 60)
    
    output_dir = project_root / "results" / "feature_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df, features = load_data(sample_size=200000)
    print(f"\nAnalyzing {len(features)} features: {features[:5]}...")
    
    corr_matrix = compute_correlation_matrix(df, features)
    
    redundant = find_redundant_pairs(corr_matrix, threshold=0.95)
    
    moderate = find_redundant_pairs(corr_matrix, threshold=0.85)
    moderate = [p for p in moderate if p['Abs_Correlation'] <= 0.95]
    if moderate:
        print(f"\n       Moderately correlated pairs (0.85 < |r| ≤ 0.95):")
        for p in moderate:
            print(f"       • {p['Feature_A']} ↔ {p['Feature_B']}: r={p['Correlation']:.4f}")
    
    importances = compute_feature_importance(df, features)
    
    kept, dropped, reasons = recommend_pruning(redundant, importances)
    
    print(f"\n       Generating plots...")
    plot_correlation_heatmap(corr_matrix, output_dir / "feature_correlation_heatmap.png")
    plot_feature_importance(importances, output_dir / "feature_importance_ranking.png")
    
    save_report(importances, redundant, kept, dropped, reasons, output_dir / "pruning_report.txt")
    
    print(f"\n{'=' * 60}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Results saved to: {output_dir}")
    print(f"  Original: {len(features)} features → Recommended: {len(kept)} features")
    print(f"{'=' * 60}")
    
    return kept, dropped


if __name__ == "__main__":
    main()

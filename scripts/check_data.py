import torch
from collections import Counter
import sys

sys.path.append('src')
try:
    import config
    from sessionization import load_sequences
    from train import load_splits
except ImportError:
    print("Error importing modules. Make sure you run this from project root.")
    sys.exit(1)

def check_distribution():
    print("Loading sequences...")
    try:
        sequences, labels, meta = load_sequences()
    except FileNotFoundError:
        print("Sequences file not found.")
        return

    print(f"Total sequences: {len(labels)}")
    
    label_counts = Counter(labels.numpy())
    print("\nOverall Class Distribution:")
    for cls_id, count in label_counts.items():
        name = config.id_to_meta_label(cls_id)
        print(f"  {cls_id} ({name}): {count}")
        
    print("\nLoading splits...")
    try:
        splits = load_splits(labels)
    except Exception as e:
        print(f"Error loading splits: {e}")
        return

    test_idx = splits['test']
    print(f"\nTest Set Size: {len(test_idx)}")
    
    test_labels = labels[test_idx].numpy()
    test_counts = Counter(test_labels)
    
    print("Test Set Class Distribution:")
    found_sqli = False
    for cls_id, count in test_counts.items():
        name = config.id_to_meta_label(cls_id)
        print(f"  {cls_id} ({name}): {count}")
        if name == "Web/SQLi":
            found_sqli = True
            
    if not found_sqli:
        print("\n[ALERT] No 'Web/SQLi' samples found in test set!")
    else:
        print("\n'Web/SQLi' samples are present.")

if __name__ == "__main__":
    check_distribution()

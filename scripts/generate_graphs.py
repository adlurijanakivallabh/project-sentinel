import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

eval_path = Path('results/evaluation_results.txt')
with open(eval_path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

cm_started = False
cm_data = []
class_names = []
for line in text.split('\n'):
    if "Confusion Matrix:" in line:
        cm_started = True
        continue
    if cm_started:
        if not line.strip():
            continue
        if len(class_names) == 0:
            class_names = ['Normal', 'DDoS/DoS', 'PortScan', 'Web/SQLi', 'Malware']
            continue
        
        parts = line.strip().split()
        if len(parts) >= 6:
            try:
                nums = [int(x) for x in parts[-5:]]
                cm_data.append(nums)
            except ValueError:
                pass
        
        if len(cm_data) == 5:
            break

if len(cm_data) == 5:
    cm = np.array(cm_data)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})
    plt.title('Test Set Confusion Matrix', pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('Ground Truth (Actual)', fontsize=14, labelpad=10)
    plt.xlabel('CNN-LSTM Prediction', fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig('results/fpga/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved confusion_matrix.png")
else:
    print("Could not find confusion matrix data.")

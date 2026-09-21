import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class_names = ['Normal', 'DDoS/DoS', 'PortScan', 'Web/SQLi', 'Malware']
cm = np.array([
    [9875, 12, 8, 0, 28],
    [2, 4091, 17, 0, 28],
    [20, 0, 994, 0, 10],
    [0, 0, 0, 1034, 0],
    [70, 5, 18, 0, 3348]
])

plt.figure(figsize=(10, 8))
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 13, "weight": "bold"},
            linewidths=1, linecolor='white', square=True)

plt.title('CNN-LSTM Confusion Matrix\n(Test Set - 19,560 Sequences)\n', fontsize=18, fontweight='bold', pad=15)
plt.ylabel('Ground Truth (Actual Attack Form)', fontsize=14, fontweight='bold', labelpad=15)
plt.xlabel('Prediction by Model', fontsize=14, fontweight='bold', labelpad=15)

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()

plt.savefig('results/fpga/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Successfully generated visually pleasing CM image at results/fpga/confusion_matrix.png")

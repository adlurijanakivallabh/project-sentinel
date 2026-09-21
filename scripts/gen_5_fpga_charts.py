import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = r"C:\Users\Janaki\Desktop\project2"

desktop_color = '#2563EB'
pynq_color = '#10B981'
bar_width = 0.35

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linestyle='--', zorder=0)
    ax.tick_params(labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('serif')

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.25)
    plt.close(fig)
    print(f"  [OK] {name} ({os.path.getsize(path)/1024:.0f} KB)")

fig1, ax1 = plt.subplots(figsize=(6, 5))
fig1.patch.set_facecolor('white')

x = np.array([0, 1])
labels = ['Desktop\n(RTX 3080, FP32)', 'PYNQ-Z2\n(ARM32, INT8)']
vals = [99.68, 99.40]
colors = [desktop_color, pynq_color]

bars = ax1.bar(x, vals, 0.5, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
ax1.set_ylim(98.0, 100.2)
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', fontfamily='serif')
ax1.set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold', 
              fontfamily='serif', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11, fontfamily='serif')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
style_ax(ax1)

for bar, val in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=13,
             fontweight='bold', fontfamily='serif', color=bar.get_facecolor())

save(fig1, 'fpga_1_accuracy.png')

fig2, ax2 = plt.subplots(figsize=(6, 5))
fig2.patch.set_facecolor('white')

vals2 = [0.9941, 0.9921]
bars2 = ax2.bar(x, vals2, 0.5, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
ax2.set_ylim(0.980, 1.002)
ax2.set_ylabel('F1-Score', fontsize=13, fontweight='bold', fontfamily='serif')
ax2.set_title('Weighted F1-Score Comparison', fontsize=14, fontweight='bold',
              fontfamily='serif', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=11, fontfamily='serif')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))
style_ax(ax2)

for bar, val in zip(bars2, vals2):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0004,
             f'{val:.4f}', ha='center', va='bottom', fontsize=13,
             fontweight='bold', fontfamily='serif', color=bar.get_facecolor())

save(fig2, 'fpga_2_f1score.png')

fig3, ax3 = plt.subplots(figsize=(6, 5))
fig3.patch.set_facecolor('white')

vals3 = [4.98, 1.34]
bars3 = ax3.bar(x, vals3, 0.5, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
ax3.set_ylim(0, 6.5)
ax3.set_ylabel('Model Size (MB)', fontsize=13, fontweight='bold', fontfamily='serif')
ax3.set_title('Model Size: FP32 vs INT8 Quantized', fontsize=14, fontweight='bold',
              fontfamily='serif', pad=12)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=11, fontfamily='serif')
style_ax(ax3)

for bar, val in zip(bars3, vals3):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f} MB', ha='center', va='bottom', fontsize=13,
             fontweight='bold', fontfamily='serif', color=bar.get_facecolor())


save(fig3, 'fpga_3_model_size.png')

fig4, ax4 = plt.subplots(figsize=(6, 5))
fig4.patch.set_facecolor('white')

vals4 = [3.1, 1800]
bars4 = ax4.bar(x, vals4, 0.5, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
ax4.set_yscale('log')
ax4.set_ylim(1, 5000)
ax4.set_ylabel('Latency (ms, log scale)', fontsize=13, fontweight='bold', fontfamily='serif')
ax4.set_title('Average Inference Latency', fontsize=14, fontweight='bold',
              fontfamily='serif', pad=12)
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=11, fontfamily='serif')
style_ax(ax4)

for bar, val in zip(bars4, vals4):
    label = f'{val:.1f} ms' if val < 100 else f'{int(val)} ms'
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
             label, ha='center', va='bottom', fontsize=13,
             fontweight='bold', fontfamily='serif', color=bar.get_facecolor())

save(fig4, 'fpga_4_latency.png')

fig5, ax5 = plt.subplots(figsize=(6, 5))
fig5.patch.set_facecolor('white')

vals5 = [250, 5]
bars5 = ax5.bar(x, vals5, 0.5, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
ax5.set_ylim(0, 310)
ax5.set_ylabel('Power Consumption (W)', fontsize=13, fontweight='bold', fontfamily='serif')
ax5.set_title('Power Consumption Comparison', fontsize=14, fontweight='bold',
              fontfamily='serif', pad=12)
ax5.set_xticks(x)
ax5.set_xticklabels(labels, fontsize=11, fontfamily='serif')
style_ax(ax5)

for bar, val in zip(bars5, vals5):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{int(val)} W', ha='center', va='bottom', fontsize=13,
             fontweight='bold', fontfamily='serif', color=bar.get_facecolor())


save(fig5, 'fpga_5_power.png')

print("\n[OK] All 5 FPGA deployment diagrams generated!")

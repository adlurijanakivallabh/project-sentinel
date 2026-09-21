import os
"""
Generate a professional accuracy bar graph for Chapter 6: Desktop vs PYNQ-Z2 comparison.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

categories = ['Accuracy\n(%)', 'F1-Score', 'Model Size\n(MB)', 'Avg Latency\n(ms)', 'Power\n(W)']

desktop_vals =  [99.68,  0.9941, 4.98,   3.1,   250]
pynq_vals =     [99.40,  0.9921, 1.34,   1800,  5]

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), gridspec_kw={'width_ratios': [2.5, 1.5, 2]})
fig.patch.set_facecolor('white')

desktop_color = '#2563EB'
pynq_color = '#10B981'
bar_width = 0.32

ax1 = axes[0]
x1 = np.array([0, 1])
labels1 = ['Accuracy (%)', 'F1-Score (×100)']
d1 = [99.68, 99.41]
p1 = [99.40, 99.21]

bars1_d = ax1.bar(x1 - bar_width/2, d1, bar_width, label='Desktop (RTX 3080)', 
                   color=desktop_color, edgecolor='white', linewidth=0.5, zorder=3)
bars1_p = ax1.bar(x1 + bar_width/2, p1, bar_width, label='PYNQ-Z2 (ARM32)',
                   color=pynq_color, edgecolor='white', linewidth=0.5, zorder=3)

ax1.set_ylim(98.5, 100.0)
ax1.set_ylabel('Performance (%)', fontsize=11, fontweight='bold', fontfamily='serif')
ax1.set_title('Classification Performance', fontsize=12, fontweight='bold', fontfamily='serif', pad=10)
ax1.set_xticks(x1)
ax1.set_xticklabels(labels1, fontsize=10, fontfamily='serif')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
ax1.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar in bars1_d:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9,
             fontweight='bold', fontfamily='serif', color=desktop_color)
for bar in bars1_p:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9,
             fontweight='bold', fontfamily='serif', color=pynq_color)


ax2 = axes[1]
x2 = np.array([0])
bars2_d = ax2.bar(x2 - bar_width/2, [4.98], bar_width, color=desktop_color, 
                   edgecolor='white', linewidth=0.5, zorder=3)
bars2_p = ax2.bar(x2 + bar_width/2, [1.34], bar_width, color=pynq_color,
                   edgecolor='white', linewidth=0.5, zorder=3)

ax2.set_ylim(0, 6.5)
ax2.set_ylabel('Size (MB)', fontsize=11, fontweight='bold', fontfamily='serif')
ax2.set_title('Model Size', fontsize=12, fontweight='bold', fontfamily='serif', pad=10)
ax2.set_xticks(x2)
ax2.set_xticklabels(['FP32 vs INT8'], fontsize=10, fontfamily='serif')
ax2.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.text(bars2_d[0].get_x() + bars2_d[0].get_width()/2, bars2_d[0].get_height() + 0.1,
         '4.98 MB', ha='center', va='bottom', fontsize=9, fontweight='bold', 
         fontfamily='serif', color=desktop_color)
ax2.text(bars2_p[0].get_x() + bars2_p[0].get_width()/2, bars2_p[0].get_height() + 0.1,
         '1.34 MB', ha='center', va='bottom', fontsize=9, fontweight='bold',
         fontfamily='serif', color=pynq_color)

ax2.annotate('73.1%\nsmaller', xy=(bar_width/2, 1.34), xytext=(0.55, 3.5),
             fontsize=9, fontweight='bold', fontfamily='serif', color='#EF4444',
             arrowprops=dict(arrowstyle='->', color='#EF4444', lw=1.5),
             ha='center')

ax3 = axes[2]
x3 = np.array([0, 1])
labels3 = ['Inference Latency\n(ms)', 'Power\n(W)']
d3 = [3.1, 250]
p3 = [1800, 5]

bars3_d = ax3.bar(x3 - bar_width/2, d3, bar_width, color=desktop_color,
                   edgecolor='white', linewidth=0.5, zorder=3)
bars3_p = ax3.bar(x3 + bar_width/2, p3, bar_width, color=pynq_color,
                   edgecolor='white', linewidth=0.5, zorder=3)

ax3.set_yscale('log')
ax3.set_ylim(1, 5000)
ax3.set_ylabel('Value (log scale)', fontsize=11, fontweight='bold', fontfamily='serif')
ax3.set_title('Latency & Power', fontsize=12, fontweight='bold', fontfamily='serif', pad=10)
ax3.set_xticks(x3)
ax3.set_xticklabels(labels3, fontsize=10, fontfamily='serif')
ax3.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

for bar, val in zip(bars3_d, d3):
    label = f'{val:.1f} ms' if val < 10 else f'{int(val)}W'
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
             label, ha='center', va='bottom', fontsize=8.5,
             fontweight='bold', fontfamily='serif', color=desktop_color)
for bar, val in zip(bars3_p, p3):
    label = f'{int(val)} ms' if val > 10 else f'{int(val)}W'
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
             label, ha='center', va='bottom', fontsize=8.5,
             fontweight='bold', fontfamily='serif', color=pynq_color)

ax3.annotate('50× less\npower', xy=(1 + bar_width/2, 5), xytext=(1.5, 60),
             fontsize=8.5, fontweight='bold', fontfamily='serif', color='#10B981',
             arrowprops=dict(arrowstyle='->', color='#10B981', lw=1.5),
             ha='center')

handles = [
    plt.Rectangle((0,0), 1, 1, facecolor=desktop_color, edgecolor='white', label='Desktop (RTX 3080, FP32)'),
    plt.Rectangle((0,0), 1, 1, facecolor=pynq_color, edgecolor='white', label='PYNQ-Z2 (ARM32, INT8)'),
]
fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=11,
           frameon=True, fancybox=True, shadow=False, edgecolor='#E5E7EB',
           prop={'family': 'serif', 'weight': 'bold'},
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Desktop vs PYNQ-Z2 FPGA Benchmark Comparison',
             fontsize=14, fontweight='bold', fontfamily='serif', y=1.02)

plt.tight_layout(rect=[0, 0.06, 1, 0.98])

out_path = r"C:\Users\Janaki\Desktop\project2\fpga_benchmark_chart.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', 
            pad_inches=0.3)
plt.close()

print(f"[OK] Saved: {out_path}")
print(f"   Size: {os.path.getsize(out_path)/1024:.0f} KB")

import os

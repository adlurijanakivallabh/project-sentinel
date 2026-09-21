import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'slide_visuals')
os.makedirs(OUT_DIR, exist_ok=True)

def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.set_xlim(-0.5, 17)
    ax.set_ylim(-1.5, 4)
    ax.axis('off')
    ax.set_title('CNN-LSTM Model Architecture for Intrusion Detection', fontsize=16, fontweight='bold', pad=15)

    blocks = [
        (0.0, 1.0, 2.0, 2.0, '#2ecc71', 'INPUT\n(20×25)\nFlow Matrix', 10),
        (2.8, 0.8, 2.4, 2.4, '#3498db', 'Conv2D(32)\n3×3 filters\nBN → ReLU\nMaxPool(2×2)', 9),
        (5.8, 0.8, 2.4, 2.4, '#2980b9', 'Conv2D(64)\n3×3 filters\nBN → ReLU\nMaxPool(2×2)', 9),
        (8.8, 0.8, 1.8, 2.4, '#9b59b6', 'Reshape\n&\nFlatten', 10),
        (11.2, 0.5, 2.4, 3.0, '#e67e22', 'LSTM\nhidden=128\n20 time\nsteps', 10),
        (14.2, 0.8, 2.4, 2.4, '#e74c3c', 'FC Layer\n→ Softmax\n5 Classes', 10),
    ]

    for x, y, w, h, color, label, fs in blocks:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='white')

    arrows = [(2.0, 2.0, 2.8, 2.0), (5.2, 2.0, 5.8, 2.0),
              (8.2, 2.0, 8.8, 2.0), (10.6, 2.0, 11.2, 2.0), (13.6, 2.0, 14.2, 2.0)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.text(4.0, -0.0, 'CNN Stage — Spatial Feature Extraction',
            ha='center', fontsize=11, fontstyle='italic', color='#2980b9')
    ax.text(11.2, -0.3, 'LSTM Stage\nTemporal Patterns',
            ha='center', fontsize=10, fontstyle='italic', color='#e67e22')

    classes = ['Normal', 'DDoS/DoS', 'PortScan', 'Web/SQLi', 'Malware']
    colors_c = ['#27ae60', '#c0392b', '#8e44ad', '#d35400', '#2c3e50']
    for i, (cls, clr) in enumerate(zip(classes, colors_c)):
        y_pos = 3.8 - i * 0.7
        ax.annotate('', xy=(17.0, y_pos), xytext=(16.6, 2.0),
                    arrowprops=dict(arrowstyle='->', color=clr, lw=1.5))
        ax.text(17.1, y_pos, cls, fontsize=9, fontweight='bold', color=clr, va='center')

    dims = [('20×25×1', 1.0, 3.3), ('18×23×32', 3.8, 3.5), ('6×9×64', 7.0, 3.5),
            ('128', 9.7, 3.5), ('128', 12.4, 3.8), ('5', 15.4, 3.5)]
    for txt, x, y in dims:
        ax.text(x, y, txt, ha='center', fontsize=8, color='#555',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#ecf0f1', edgecolor='#bdc3c7'))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'cnn_lstm_architecture.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


def draw_data_representation():
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={'width_ratios': [1, 1.2, 1]})

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('① Raw Network Packets', fontsize=13, fontweight='bold')

    packets = [
        ('Packet 1', 'TCP SYN | 192.168.1.5 → 10.0.0.1:80', '#3498db'),
        ('Packet 2', 'TCP ACK | 192.168.1.5 → 10.0.0.1:80', '#3498db'),
        ('Packet 3', 'HTTP GET /login?id=1 OR 1=1', '#e74c3c'),
        ('Packet 4', 'TCP SYN | 192.168.1.5 → 10.0.0.1:443', '#3498db'),
        ('Packet 5', 'UDP Flood | 10.0.0.5 → 10.0.0.1:53', '#e74c3c'),
    ]
    for i, (name, desc, clr) in enumerate(packets):
        y = 10.5 - i * 2
        rect = FancyBboxPatch((0.5, y - 0.6), 9, 1.2, boxstyle="round,pad=0.1",
                              facecolor=clr, alpha=0.15, edgecolor=clr, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(1.0, y + 0.1, name, fontsize=9, fontweight='bold', color=clr)
        ax.text(1.0, y - 0.3, desc, fontsize=7.5, color='#333')

    ax.annotate('', xy=(5, 0.2), xytext=(5, 0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(5, -0.1, 'Feature Extraction →', ha='center', fontsize=10, fontweight='bold')

    ax = axes[1]
    ax.axis('off')
    ax.set_title('② 25 Flow Features Extracted', fontsize=13, fontweight='bold')

    features = [
        'Duration', 'Protocol', 'Src Port', 'Dst Port', 'Fwd Packets',
        'Bwd Packets', 'Fwd Pkt Len', 'Bwd Pkt Len', 'Flow Bytes/s', 'Flow Pkts/s',
        'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd PSH Flags',
        'Bwd PSH Flags', 'Fwd Header Len', 'Bwd Header Len', 'Pkt Len Mean', 'Pkt Len Std',
        'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'ACK Flag Cnt', 'Down/Up Ratio',
    ]
    for i, feat in enumerate(features):
        row, col = i // 5, i % 5
        x, y = col * 2.0 + 0.2, 10.5 - row * 1.8
        rect = FancyBboxPatch((x, y - 0.5), 1.8, 1.0, boxstyle="round,pad=0.05",
                              facecolor='#3498db', alpha=0.2, edgecolor='#2980b9')
        ax.add_patch(rect)
        ax.text(x + 0.9, y, feat, ha='center', va='center', fontsize=6.5, fontweight='bold')
    ax.set_xlim(-0.2, 10.5)
    ax.set_ylim(-0.5, 12)

    ax = axes[2]
    ax.set_title('③ Sliding Window (20×25)', fontsize=13, fontweight='bold')

    np.random.seed(42)
    data = np.random.rand(20, 25)
    data[5:10, 8:12] = 0.9
    data[15:18, 0:3] = 0.1

    im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    ax.set_xlabel('25 Features →', fontsize=10)
    ax.set_ylabel('← 20 Time Steps', fontsize=10)
    ax.set_xticks([0, 12, 24])
    ax.set_xticklabels(['F1', 'F13', 'F25'], fontsize=8)
    ax.set_yticks([0, 9, 19])
    ax.set_yticklabels(['t₁', 't₁₀', 't₂₀'], fontsize=8)

    rect = plt.Rectangle((- 0.5, -0.5), 25, 20, linewidth=3, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(12.5, -2, 'Window=20, Step=10', ha='center', fontsize=9, fontweight='bold', color='red')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'data_representation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


def draw_cnn_feature_extraction():
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(-1, 16)
    ax.set_ylim(-2, 10)
    ax.axis('off')
    ax.set_title('CNN Spatial Feature Extraction from Network Traffic', fontsize=15, fontweight='bold', pad=15)

    np.random.seed(42)
    input_ax = fig.add_axes([0.02, 0.25, 0.15, 0.5])
    data = np.random.rand(20, 25)
    data[5:10, 10:15] = 0.95
    input_ax.imshow(data, cmap='Blues', aspect='auto')
    input_ax.set_title('Input\n20×25×1', fontsize=9, fontweight='bold')
    input_ax.set_xticks([])
    input_ax.set_yticks([])

    conv1_ax = fig.add_axes([0.22, 0.2, 0.15, 0.55])
    d1 = np.random.rand(18, 32)
    conv1_ax.imshow(d1, cmap='Oranges', aspect='auto')
    conv1_ax.set_title('After Conv1+Pool\n9×11×32', fontsize=9, fontweight='bold')
    conv1_ax.set_xticks([])
    conv1_ax.set_yticks([])

    conv2_ax = fig.add_axes([0.42, 0.25, 0.12, 0.45])
    d2 = np.random.rand(4, 64)
    conv2_ax.imshow(d2, cmap='Purples', aspect='auto')
    conv2_ax.set_title('After Conv2+Pool\n4×4×64', fontsize=9, fontweight='bold')
    conv2_ax.set_xticks([])
    conv2_ax.set_yticks([])

    flat_ax = fig.add_axes([0.58, 0.35, 0.03, 0.3])
    flat_data = np.random.rand(128, 1)
    flat_ax.imshow(flat_data, cmap='Greens', aspect='auto')
    flat_ax.set_title('Flat\n1024', fontsize=8, fontweight='bold')
    flat_ax.set_xticks([])
    flat_ax.set_yticks([])

    lstm_ax = fig.add_axes([0.66, 0.3, 0.12, 0.35])
    lstm_data = np.random.rand(20, 1)
    lstm_ax.imshow(lstm_data, cmap='YlOrRd', aspect='auto')
    lstm_ax.set_title('LSTM\nhidden=128', fontsize=9, fontweight='bold')
    lstm_ax.set_xticks([])
    lstm_ax.set_yticks([])
    lstm_ax.set_ylabel('20 steps', fontsize=8)

    out_ax = fig.add_axes([0.83, 0.35, 0.12, 0.3])
    classes = ['Normal', 'DDoS', 'PortScan', 'SQLi', 'Malware']
    probs = [0.02, 0.95, 0.01, 0.01, 0.01]
    colors = ['#27ae60', '#e74c3c', '#8e44ad', '#d35400', '#2c3e50']
    out_ax.barh(classes, probs, color=colors, edgecolor='black')
    out_ax.set_title('Softmax\nOutput', fontsize=9, fontweight='bold')
    out_ax.set_xlim(0, 1.0)
    out_ax.tick_params(labelsize=7)

    flow_labels = [
        (0.19, 0.50, '→', 20), (0.39, 0.50, '→', 20),
        (0.56, 0.50, '→', 20), (0.63, 0.50, '→', 20),
        (0.80, 0.50, '→', 20),
    ]
    for x, y, txt, fs in flow_labels:
        fig.text(x, y, txt, fontsize=fs, fontweight='bold', ha='center', va='center')

    descs = [
        (0.19, 0.12, 'Conv2D(32, 3×3)\nBatchNorm\nReLU, MaxPool'),
        (0.39, 0.12, 'Conv2D(64, 3×3)\nBatchNorm\nReLU, MaxPool'),
        (0.55, 0.12, 'Reshape\n→ Flatten'),
        (0.72, 0.12, 'LSTM(128)\nTemporal\nEncoding'),
        (0.88, 0.12, 'FC → Softmax\n5 Classes'),
    ]
    for x, y, txt in descs:
        fig.text(x, y, txt, fontsize=8, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor='#bdc3c7'))

    path = os.path.join(OUT_DIR, 'cnn_feature_extraction.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


def draw_formulas():
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('CNN — Convolution & Pooling Formulas', fontsize=14, fontweight='bold')

    cnn_formulas = [
        (5, 8.5, r'$\mathbf{Convolution:}\ \ Z^{(l)}_{i,j,k} = \sum_{m}\sum_{n}\sum_{c} W^{(l)}_{m,n,c,k} \cdot X^{(l-1)}_{i+m,\, j+n,\, c} + b^{(l)}_k$', 13),
        (5, 7.0, r'$\mathbf{BatchNorm:}\ \ \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$', 13),
        (5, 5.5, r'$\mathbf{ReLU\ Activation:}\ \ f(x) = \max(0, x)$', 13),
        (5, 4.0, r'$\mathbf{Max\ Pooling:}\ \ P_{i,j} = \max_{(m,n) \in R_{i,j}} Z_{m,n}$', 13),
        (5, 2.5, r'$\mathbf{Dimensions:}\ (20{\times}25{\times}1) \rightarrow (18{\times}23{\times}32) \rightarrow (9{\times}11{\times}32) \rightarrow (7{\times}9{\times}64) \rightarrow (3{\times}4{\times}64)$', 11),
    ]
    for x, y, formula, fs in cnn_formulas:
        ax.text(x, y, formula, fontsize=fs, ha='center', va='center')

    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('LSTM — Gate Equations', fontsize=14, fontweight='bold')

    lstm_formulas = [
        (5, 8.5, r'$\mathbf{Forget\ Gate:}\ \ f_t = \sigma(W_f \cdot [h_{t-1},\, x_t] + b_f)$', 13),
        (5, 7.0, r'$\mathbf{Input\ Gate:}\ \ i_t = \sigma(W_i \cdot [h_{t-1},\, x_t] + b_i)$', 13),
        (5, 5.5, r'$\mathbf{Candidate:}\ \ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1},\, x_t] + b_C)$', 13),
        (5, 4.0, r'$\mathbf{Cell\ State:}\ \ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$', 13),
        (5, 2.5, r'$\mathbf{Output\ Gate:}\ \ o_t = \sigma(W_o \cdot [h_{t-1},\, x_t] + b_o)$', 13),
        (5, 1.0, r'$\mathbf{Hidden\ State:}\ \ h_t = o_t \odot \tanh(C_t)$', 13),
    ]
    for x, y, formula, fs in lstm_formulas:
        ax.text(x, y, formula, fontsize=fs, ha='center', va='center')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'mathematical_formulas.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


def draw_lstm_cell():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 9)
    ax.axis('off')
    ax.set_title('LSTM Cell Architecture — Temporal Pattern Recognition', fontsize=14, fontweight='bold', pad=15)

    ax.annotate('', xy=(11, 7.5), xytext=(1, 7.5),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))
    ax.text(0.2, 7.5, r'$C_{t-1}$', fontsize=14, fontweight='bold', color='#e74c3c')
    ax.text(11.3, 7.5, r'$C_t$', fontsize=14, fontweight='bold', color='#e74c3c')
    ax.text(6, 8.2, 'Cell State', fontsize=11, ha='center', fontstyle='italic', color='#e74c3c')

    ax.annotate('', xy=(11, 2), xytext=(1, 2),
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=3))
    ax.text(0.2, 2, r'$h_{t-1}$', fontsize=14, fontweight='bold', color='#2980b9')
    ax.text(11.3, 2, r'$h_t$', fontsize=14, fontweight='bold', color='#2980b9')

    gates = [
        (3, 5, 'Forget\nGate\n$f_t$', '#e67e22'),
        (5.5, 5, 'Input\nGate\n$i_t$', '#27ae60'),
        (7.5, 5, 'Candidate\n$\\tilde{C}_t$', '#8e44ad'),
        (9.5, 5, 'Output\nGate\n$o_t$', '#3498db'),
    ]
    for x, y, label, color in gates:
        circle = plt.Circle((x, y), 0.9, facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.text(6, -0.3, r'$x_t$ (Input at time t)', fontsize=12, ha='center', fontweight='bold', color='#2c3e50')
    for gx in [3, 5.5, 7.5, 9.5]:
        ax.annotate('', xy=(gx, 4.1), xytext=(gx, 0.5),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2))

    ax.annotate('', xy=(3, 7.5), xytext=(3, 5.9),
                arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2))
    ax.annotate('', xy=(6.5, 7.5), xytext=(5.5, 5.9),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    ax.annotate('', xy=(6.5, 7.5), xytext=(7.5, 5.9),
                arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2))
    ax.annotate('', xy=(9.5, 2), xytext=(9.5, 4.1),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))

    ops = [
        (3, 7.5, '×', '#e67e22', 12),
        (6.5, 7.5, '+', '#27ae60', 12),
        (9.5, 3.5, 'tanh', '#3498db', 9),
    ]
    for x, y, sym, color, fs in ops:
        ax.text(x, y, sym, fontsize=fs, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor=color, linewidth=2))

    ax.text(0.5, -0.8, 'σ = Sigmoid | tanh = Hyperbolic Tangent | ⊙ = Element-wise Multiply | + = Addition',
            fontsize=10, color='#555')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'lstm_cell_diagram.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


def draw_feature_extraction_explained():
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle('How CNN-LSTM Extracts Attack Features from Network Traffic', fontsize=15, fontweight='bold')

    ax = axes[0]
    np.random.seed(10)
    normal = np.random.uniform(0.3, 0.5, (20, 25))
    ax.imshow(normal, cmap='Greens', vmin=0, vmax=1, aspect='auto')
    ax.set_title('Normal Traffic\n(Low variance, uniform)', fontsize=10, fontweight='bold', color='green')
    ax.set_xlabel('25 Features')
    ax.set_ylabel('20 Time Steps')

    ax = axes[1]
    ddos = np.random.uniform(0.2, 0.4, (20, 25))
    ddos[:, 4:6] = np.linspace(0.3, 1.0, 20).reshape(-1, 1)
    ddos[:, 8:10] = 0.95
    ddos[:, 0] = 0.05
    ax.imshow(ddos, cmap='Reds', vmin=0, vmax=1, aspect='auto')
    ax.set_title('DDoS Attack\n(High pkt rate, short duration)', fontsize=10, fontweight='bold', color='red')
    ax.set_xlabel('25 Features')

    ax = axes[2]
    portscan = np.random.uniform(0.2, 0.4, (20, 25))
    for i in range(20):
        portscan[i, 3] = i / 20.0
    portscan[:, 4] = 0.1
    portscan[:, 0] = 0.05
    portscan[:, 20:24] = 0.9
    ax.imshow(portscan, cmap='Purples', vmin=0, vmax=1, aspect='auto')
    ax.set_title('PortScan\n(Incremental ports, SYN flags)', fontsize=10, fontweight='bold', color='purple')
    ax.set_xlabel('25 Features')

    ax = axes[3]
    sqli = np.random.uniform(0.3, 0.5, (20, 25))
    sqli[:, 3] = 0.32
    sqli[:, 6:8] = np.random.uniform(0.7, 1.0, (20, 2))
    sqli[::3, :] = sqli[::3, :] + 0.2
    sqli = np.clip(sqli, 0, 1)
    ax.imshow(sqli, cmap='Oranges', vmin=0, vmax=1, aspect='auto')
    ax.set_title('SQL Injection\n(HTTP port, large payload, repetition)', fontsize=10, fontweight='bold', color='darkorange')
    ax.set_xlabel('25 Features')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'feature_extraction_explained.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    print("Generating Slide 6 visuals...\n")
    draw_architecture()
    draw_data_representation()
    draw_cnn_feature_extraction()
    draw_formulas()
    draw_lstm_cell()
    draw_feature_extraction_explained()
    print(f"\nAll images saved to: {OUT_DIR}")

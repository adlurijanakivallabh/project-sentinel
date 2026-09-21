import numpy as np
import time
import os
import sys
import argparse

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sentinel_numpy import SentinelNumPy

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, 'sentinel_weights.npz')
ZSCORE = os.path.join(BASE, 'zscore_stats.npz')
SAMPLES = os.path.join(BASE, 'traffic_samples.npz')

CLASS_NAMES = ['Normal', 'DoS/DDoS', 'PortScan/Recon', 'Web/Injection',
               'Brute Force', 'Botnet/C2', 'Malware/Exploit', 'Infiltration']

FEAT_SHORT = [
    'DstPort', 'Proto', 'Duration', 'FwdPkts', 'BwdPkts',
    'FwdLen', 'BwdLen', 'FwdMax', 'FwdMin', 'FwdMean',
    'FwdStd', 'BwdMax', 'BwdMin', 'BwdMean', 'BwdStd',
    'IATMean', 'IATStd', 'IATMax', 'IATMin', 'FwdPSH',
    'BwdPSH', 'FwdURG', 'BwdURG', 'Bytes/s', 'Pkts/s',
    'dBytes', 'dPkts', 'rollFwd', 'rollBwd', 'varDur',
]

SEVERITY_MAP = {
    0: ('SAFE', '\033[92m'),
    1: ('CRIT', '\033[91m'),
    2: ('WARN', '\033[93m'),
    3: ('CRIT', '\033[91m'),
    4: ('HIGH', '\033[91m'),
    5: ('CRIT', '\033[91m'),
    6: ('CRIT', '\033[91m'),
    7: ('HIGH', '\033[91m'),
}
RST = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'


def softmax_t(logits, temp=0.5):
    s = logits / temp
    e = np.exp(s - s.max())
    return e / e.sum()


def fmt_val(v):
    if abs(v) >= 1000:
        return f'{v:>8.0f}'
    if abs(v) >= 1:
        return f'{v:>8.2f}'
    return f'{v:>8.4f}'


def show_sample(sample, label, idx=0):
    print(f'\n{DIM}{"─"*70}')
    print(f'  INPUT: {label} (sample #{idx})  |  Shape: {sample.shape}{RST}')
    print(f'{DIM}{"─"*70}{RST}')

    key_feats = [0, 2, 3, 4, 23, 24, 25, 26]
    header = '  Flow  ' + ''.join(f'{FEAT_SHORT[f]:>9s}' for f in key_feats)
    print(f'{DIM}{header}{RST}')

    rows_to_show = [0, 1, 2, -2, -1]
    for ri, r in enumerate(rows_to_show):
        if ri == 3:
            print(f'{DIM}  ...   {"":>72s}{RST}')
        flow_num = r if r >= 0 else sample.shape[0] + r
        vals = ''.join(fmt_val(sample[r, f]) for f in key_feats)
        print(f'  F{flow_num+1:<3d}  {vals}')

    nz = np.count_nonzero(sample)
    print(f'{DIM}  Stats: min={sample.min():.3f}  max={sample.max():.3f}  '
          f'mean={sample.mean():.3f}  nonzero={nz}/{sample.size}{RST}')


def show_prediction(logits, true_label, elapsed_ms):
    probs = softmax_t(logits[0])
    pred_id = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_id]
    conf = probs[pred_id] * 100
    correct = pred_name == true_label

    sev_name, sev_col = SEVERITY_MAP.get(pred_id, ('?', ''))

    status = f'{BOLD}\033[92m✓ CORRECT{RST}' if correct else f'{BOLD}\033[91m✗ WRONG{RST}'
    print(f'\n  {BOLD}PREDICTION:{RST} {sev_col}{pred_name}{RST}  '
          f'({conf:.1f}%)  [{sev_name}]  {elapsed_ms:.0f}ms  {status}')

    print(f'{DIM}  {"─"*50}{RST}')
    for i, name in enumerate(CLASS_NAMES):
        p = probs[i] * 100
        bar_len = int(p / 5)
        bar = '█' * bar_len
        marker = ' ◄' if i == pred_id else ''
        col = sev_col if i == pred_id else DIM
        print(f'  {col}{name:<18s} {p:5.1f}% {bar}{marker}{RST}')

    return pred_id, conf, correct


def load_engine():
    print(f'{BOLD}{"═"*60}')
    print(f'  SENTINEL V2 — PYNQ EDGE IPS (CLI)')
    print(f'{"═"*60}{RST}')

    t0 = time.time()
    engine = SentinelNumPy(WEIGHTS, ZSCORE)
    load_time = time.time() - t0
    print(f'  Model loaded in {load_time:.1f}s')

    data = np.load(SAMPLES)
    total = sum(len(data[k]) for k in data.keys())
    print(f'  Samples: {total} windows across {len(data.keys())} classes')
    for k in CLASS_NAMES:
        if k in data:
            print(f'    {k}: {len(data[k])}')

    return engine, data


def run_single_demo(engine, data, class_filter=None):
    print(f'\n{BOLD}{"─"*60}')
    print(f'  SINGLE SAMPLE CLASSIFICATION')
    print(f'{"─"*60}{RST}')

    classes = [class_filter] if class_filter else CLASS_NAMES
    correct = 0
    total = 0
    total_ms = 0.0

    for cls in classes:
        if cls not in data:
            print(f'  [SKIP] {cls}: no samples')
            continue

        sample = data[cls][0:1].astype(np.float32)

        show_sample(sample[0], cls, 0)

        t0 = time.perf_counter()
        logits = engine.forward(sample)
        ms = (time.perf_counter() - t0) * 1000

        _, conf, ok = show_prediction(logits, cls, ms)
        if ok:
            correct += 1
        total += 1
        total_ms += ms

    if total > 0:
        print(f'\n{BOLD}{"─"*60}')
        print(f'  Summary: {correct}/{total} correct ({correct/total*100:.0f}%)  '
              f'Avg: {total_ms/total:.0f}ms/sample')
        print(f'{"─"*60}{RST}')


def run_batch_test(engine, data, n_per_class=5):
    print(f'\n{BOLD}{"═"*60}')
    print(f'  BATCH TEST ({n_per_class} samples per class)')
    print(f'{"═"*60}{RST}')
    print(f'  {"Class":<18s}  {"Acc":>5s}  {"Avg ms":>7s}  {"Conf":>6s}  Result')
    print(f'  {"─"*55}')

    grand_correct = 0
    grand_total = 0
    grand_ms = 0.0

    for cls in CLASS_NAMES:
        if cls not in data:
            continue
        n = min(n_per_class, len(data[cls]))
        cls_correct = 0
        cls_ms = 0.0
        cls_conf = 0.0

        for i in range(n):
            sample = data[cls][i:i+1].astype(np.float32)
            t0 = time.perf_counter()
            logits = engine.forward(sample)
            ms = (time.perf_counter() - t0) * 1000

            probs = softmax_t(logits[0])
            pred = int(np.argmax(probs))
            if CLASS_NAMES[pred] == cls:
                cls_correct += 1
            cls_ms += ms
            cls_conf += probs[pred] * 100

        acc = cls_correct / n * 100
        avg_ms = cls_ms / n
        avg_conf = cls_conf / n
        ok = '✓' if cls_correct == n else f'{cls_correct}/{n}'
        col = '\033[92m' if cls_correct == n else '\033[93m'
        print(f'  {col}{cls:<18s}  {acc:4.0f}%  {avg_ms:5.0f}ms  {avg_conf:5.1f}%  {ok}{RST}')

        grand_correct += cls_correct
        grand_total += n
        grand_ms += cls_ms

    print(f'  {"─"*55}')
    print(f'  {BOLD}Overall: {grand_correct}/{grand_total} '
          f'({grand_correct/grand_total*100:.1f}%)  '
          f'Avg: {grand_ms/grand_total:.0f}ms{RST}')


def run_all_test(engine, data):
    print(f'\n{BOLD}{"═"*60}')
    print(f'  FULL TEST (all samples)')
    print(f'{"═"*60}{RST}')

    grand_correct = 0
    grand_total = 0

    for cls in CLASS_NAMES:
        if cls not in data:
            continue
        samples = data[cls].astype(np.float32)
        n = len(samples)
        cls_correct = 0

        for i in range(n):
            logits = engine.forward(samples[i:i+1])
            pred = int(np.argmax(logits[0]))
            if CLASS_NAMES[pred] == cls:
                cls_correct += 1

        acc = cls_correct / n * 100
        col = '\033[92m' if acc >= 95 else '\033[93m' if acc >= 80 else '\033[91m'
        print(f'  {col}{cls:<18s}  {cls_correct:>4d}/{n:<4d}  ({acc:.1f}%){RST}')
        grand_correct += cls_correct
        grand_total += n

    print(f'  {"─"*40}')
    print(f'  {BOLD}Total: {grand_correct}/{grand_total} ({grand_correct/grand_total*100:.2f}%){RST}')


def run_benchmark(engine, data, n_iters=50):
    print(f'\n{BOLD}{"═"*60}')
    print(f'  BENCHMARK ({n_iters} iterations)')
    print(f'{"═"*60}{RST}')

    sample = data['Normal'][0:1].astype(np.float32)

    for _ in range(5):
        engine.forward(sample)

    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        engine.forward(sample)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    print(f'  Mean:       {lat.mean():.1f} ms')
    print(f'  Median:     {np.median(lat):.1f} ms')
    print(f'  Min/Max:    {lat.min():.1f} / {lat.max():.1f} ms')
    print(f'  Std:        {lat.std():.1f} ms')
    print(f'  P95:        {np.percentile(lat, 95):.1f} ms')
    print(f'  Throughput: {1000/lat.mean():.0f} windows/sec')

    batch_sizes = [1, 5, 10]
    print(f'\n  {"Batch":>5s}  {"Total ms":>8s}  {"Per-sample":>10s}  {"Flows/sec":>10s}')
    for bs in batch_sizes:
        batch = np.tile(sample, (bs, 1, 1))
        t0 = time.perf_counter()
        engine.forward(batch)
        total = (time.perf_counter() - t0) * 1000
        print(f'  {bs:>5d}  {total:>7.1f}ms  {total/bs:>8.1f}ms  {1000*bs/total:>9.0f}')


def run_confusion_matrix(engine, data, n_per_class=20):
    print(f'\n{BOLD}{"═"*60}')
    print(f'  CONFUSION MATRIX ({n_per_class} samples/class)')
    print(f'{"═"*60}{RST}')

    nc = len(CLASS_NAMES)
    matrix = np.zeros((nc, nc), dtype=int)
    short = ['Norm', 'DoS', 'Scan', 'Web', 'BF', 'Bot', 'Malw', 'Infil']

    for ci, cls in enumerate(CLASS_NAMES):
        if cls not in data:
            continue
        n = min(n_per_class, len(data[cls]))
        for i in range(n):
            sample = data[cls][i:i+1].astype(np.float32)
            logits = engine.forward(sample)
            pred = int(np.argmax(logits[0]))
            matrix[ci][pred] += 1

    header = '  True\\Pred  ' + ''.join(f'{s:>6s}' for s in short)
    print(f'{DIM}{header}{RST}')
    print(f'  {"─"*60}')

    for ci, cls in enumerate(CLASS_NAMES):
        row = f'  {short[ci]:<10s} '
        for pi in range(nc):
            val = matrix[ci][pi]
            if ci == pi and val > 0:
                row += f'{BOLD}\033[92m{val:>6d}{RST}'
            elif val > 0:
                row += f'\033[91m{val:>6d}{RST}'
            else:
                row += f'{DIM}{val:>6d}{RST}'
        total = matrix[ci].sum()
        acc = matrix[ci][ci] / total * 100 if total > 0 else 0
        row += f'  {acc:5.1f}%'
        print(row)

    total_correct = np.trace(matrix)
    total_samples = matrix.sum()
    print(f'  {"─"*60}')
    print(f'  {BOLD}Overall: {total_correct}/{total_samples} ({total_correct/total_samples*100:.1f}%){RST}')


def run_metrics(engine, data, n_per_class=20):
    print(f'\n{BOLD}{"═"*60}')
    print(f'  CLASSIFICATION METRICS ({n_per_class} samples/class)')
    print(f'{"═"*60}{RST}')

    nc = len(CLASS_NAMES)
    matrix = np.zeros((nc, nc), dtype=int)

    for ci, cls in enumerate(CLASS_NAMES):
        if cls not in data:
            continue
        n = min(n_per_class, len(data[cls]))
        for i in range(n):
            sample = data[cls][i:i+1].astype(np.float32)
            logits = engine.forward(sample)
            pred = int(np.argmax(logits[0]))
            matrix[ci][pred] += 1

    print(f'  {"Class":<18s}  {"Prec":>6s}  {"Recall":>6s}  {"F1":>6s}  {"Support":>7s}')
    print(f'  {"─"*50}')

    total_tp = 0
    total_support = 0
    weighted_p = weighted_r = weighted_f1 = 0.0

    for ci, cls in enumerate(CLASS_NAMES):
        tp = matrix[ci][ci]
        fp = matrix[:, ci].sum() - tp
        fn = matrix[ci, :].sum() - tp
        support = matrix[ci, :].sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        col = '\033[92m' if f1 >= 0.95 else '\033[93m' if f1 >= 0.8 else '\033[91m'
        print(f'  {col}{cls:<18s}  {prec*100:5.1f}%  {rec*100:5.1f}%  {f1*100:5.1f}%  {support:>7d}{RST}')

        total_tp += tp
        total_support += support
        weighted_p += prec * support
        weighted_r += rec * support
        weighted_f1 += f1 * support

    print(f'  {"─"*50}')
    if total_support > 0:
        print(f'  {BOLD}{"Weighted Avg":<18s}  '
              f'{weighted_p/total_support*100:5.1f}%  '
              f'{weighted_r/total_support*100:5.1f}%  '
              f'{weighted_f1/total_support*100:5.1f}%  '
              f'{total_support:>7d}{RST}')
        print(f'  {BOLD}Accuracy: {total_tp}/{total_support} ({total_tp/total_support*100:.2f}%){RST}')


def run_attention_analysis(engine, data):
    print(f'\n{BOLD}{"═"*60}')
    print(f'  ATTENTION ANALYSIS (which time steps matter)')
    print(f'{"═"*60}{RST}')
    print(f'{DIM}  After CNN, 20 flows compress to 5 time steps:')
    print(f'  T1=F1-F4  T2=F5-F8  T3=F9-F12  T4=F13-F16  T5=F17-F20{RST}\n')

    for cls in CLASS_NAMES:
        if cls not in data:
            continue
        n = min(10, len(data[cls]))
        attn_sum = np.zeros(5)

        for i in range(n):
            sample = data[cls][i:i+1].astype(np.float32)
            base_logits = engine.forward(sample)
            base_pred = int(np.argmax(base_logits[0]))
            base_conf = softmax_t(base_logits[0])[base_pred]

            for t in range(5):
                masked = sample.copy()
                start_flow = t * 4
                end_flow = min(start_flow + 4, 20)
                masked[0, start_flow:end_flow, :] = 0
                masked_logits = engine.forward(masked)
                masked_conf = softmax_t(masked_logits[0])[base_pred]
                attn_sum[t] += max(0, base_conf - masked_conf)

        attn_avg = attn_sum / n
        attn_norm = attn_avg / (attn_avg.sum() + 1e-8)

        sev_col = SEVERITY_MAP.get(CLASS_NAMES.index(cls), ('', ''))[1]
        bars = ''
        for t in range(5):
            bar_len = int(attn_norm[t] * 30)
            bars += f'T{t+1}:{"█" * bar_len}{"░" * (6 - bar_len)} '
        peak = int(np.argmax(attn_norm)) + 1
        print(f'  {sev_col}{cls:<18s}{RST} {bars} peak=T{peak}')

    print(f'\n{DIM}  Higher bars = model focuses more on those flows.')
    print(f'  DoS: early flows (sudden spike). Infiltration: late flows (slow movement).{RST}')


def run_feature_importance(data):
    print(f'\n{BOLD}{"═"*60}')
    print(f'  FEATURE IMPORTANCE (Normal vs Attack difference)')
    print(f'{"═"*60}{RST}')

    if 'Normal' not in data:
        print('  [SKIP] No Normal samples')
        return

    normal_mean = data['Normal'].mean(axis=(0, 1))

    for cls in CLASS_NAMES[1:]:
        if cls not in data:
            continue
        attack_mean = data[cls].mean(axis=(0, 1))
        diff = np.abs(attack_mean - normal_mean)
        top_idx = np.argsort(diff)[::-1][:5]

        sev_col = SEVERITY_MAP.get(CLASS_NAMES.index(cls), ('', ''))[1]
        top_feats = ', '.join(f'{FEAT_SHORT[i]}({diff[i]:.3f})' for i in top_idx)
        print(f'  {sev_col}{cls:<18s}{RST} Top 5: {top_feats}')

    print(f'\n{DIM}  Values show absolute z-score difference vs Normal traffic.{RST}')


def run_model_info(engine):
    print(f'\n{BOLD}{"═"*60}')
    print(f'  MODEL & SYSTEM INFO')
    print(f'{"═"*60}{RST}')

    total_params = 0
    total_bytes = 0
    layer_groups = {}

    for key, val in engine.w.items():
        params = val.size
        nbytes = val.nbytes
        total_params += params
        total_bytes += nbytes

        parts = key.split('.')
        group = parts[0] if parts[0] not in ('0', '1', '2', '3', '4') else 'cnn'
        if group not in layer_groups:
            layer_groups[group] = {'params': 0, 'bytes': 0}
        layer_groups[group]['params'] += params
        layer_groups[group]['bytes'] += nbytes

    print(f'  {BOLD}Model Architecture:{RST} SentinelV2 (MultiScale CNN-BiLSTM-GRU-MHA)')
    print(f'  {"─"*50}')
    print(f'  {"Module":<20s}  {"Params":>10s}  {"Size (KB)":>10s}  {"% Total":>8s}')
    print(f'  {"─"*50}')

    for group, info in sorted(layer_groups.items()):
        pct = info['params'] / total_params * 100
        print(f'  {group:<20s}  {info["params"]:>10,d}  {info["bytes"]/1024:>9.1f}  {pct:>7.1f}%')

    print(f'  {"─"*50}')
    print(f'  {BOLD}{"TOTAL":<20s}  {total_params:>10,d}  {total_bytes/1024:>9.1f}  100.0%{RST}')

    print(f'\n  {BOLD}Memory Usage:{RST}')
    weights_mb = total_bytes / 1024 / 1024
    samples_mb = os.path.getsize(SAMPLES) / 1024 / 1024 if os.path.exists(SAMPLES) else 0
    runtime_mb = 20 * 30 * 4 / 1024 / 1024 * 10
    print(f'    Weights (FP32):    {weights_mb:.1f} MB')
    print(f'    Weights (INT8):    {weights_mb/4:.1f} MB  (quantized)')
    print(f'    Sample cache:      {samples_mb:.1f} MB')
    print(f'    Runtime estimate:  ~{runtime_mb:.1f} MB activations')
    print(f'    Total estimate:    ~{weights_mb + samples_mb + runtime_mb:.1f} MB')
    print(f'    PYNQ-Z2 RAM:      512 MB (fits easily)')

    print(f'\n  {BOLD}PYNQ-Z2 Board:{RST}')
    print(f'    SoC:       Xilinx Zynq-7020 (XC7Z020-1CLG400C)')
    print(f'    CPU:       Dual ARM Cortex-A9 @ 650 MHz')
    print(f'    FPGA:      85K logic cells, 220 DSP slices')
    print(f'    RAM:       512 MB DDR3')
    print(f'    Power:     ~2.5W typical (vs ~300W GPU server)')
    print(f'    Inference: Pure NumPy on ARM (no FPGA overlay needed)')


def run_export(engine, data, n_per_class=20):
    out_path = os.path.join(BASE, 'results.csv')
    print(f'\n{BOLD}{"═"*60}')
    print(f'  EXPORTING RESULTS TO CSV')
    print(f'{"═"*60}{RST}')

    rows = []
    for ci, cls in enumerate(CLASS_NAMES):
        if cls not in data:
            continue
        n = min(n_per_class, len(data[cls]))
        for i in range(n):
            sample = data[cls][i:i+1].astype(np.float32)
            t0 = time.perf_counter()
            logits = engine.forward(sample)
            ms = (time.perf_counter() - t0) * 1000
            probs = softmax_t(logits[0])
            pred = int(np.argmax(probs))
            rows.append(f'{cls},{CLASS_NAMES[pred]},{probs[pred]*100:.1f},{ms:.1f},'
                        + ','.join(f'{probs[j]*100:.2f}' for j in range(8)))

    header = ('TrueClass,PredClass,Confidence,LatencyMs,'
              + ','.join(f'Prob_{c.replace("/","_").replace(" ","_")}' for c in CLASS_NAMES))

    with open(out_path, 'w') as f:
        f.write(header + '\n')
        for r in rows:
            f.write(r + '\n')

    print(f'  Saved {len(rows)} predictions to: {out_path}')
    print(f'  Columns: TrueClass, PredClass, Confidence, LatencyMs, Prob_*')


def main():
    parser = argparse.ArgumentParser(
        description='Sentinel V2 PYNQ CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 demo_cli.py                    Full demo + batch test
  python3 demo_cli.py --class DoS/DDoS  Test single class
  python3 demo_cli.py --bench 100       Benchmark latency
  python3 demo_cli.py --all             Test all samples
  python3 demo_cli.py --confusion       Confusion matrix
  python3 demo_cli.py --metrics         Precision/Recall/F1
  python3 demo_cli.py --attention       Attention analysis
  python3 demo_cli.py --features        Feature importance
  python3 demo_cli.py --info            Model & board info
  python3 demo_cli.py --export          Export results to CSV
  python3 demo_cli.py --full-report     Run everything
""")
    parser.add_argument('--class', dest='cls', help='Test specific class')
    parser.add_argument('--bench', type=int, default=0, help='Benchmark N iterations')
    parser.add_argument('--all', action='store_true', help='Test all samples')
    parser.add_argument('--batch', type=int, default=5, help='Samples per class in batch test')
    parser.add_argument('--no-demo', action='store_true', help='Skip single-sample demo')
    parser.add_argument('--confusion', action='store_true', help='Show confusion matrix')
    parser.add_argument('--metrics', action='store_true', help='Show precision/recall/F1')
    parser.add_argument('--attention', action='store_true', help='Show attention analysis')
    parser.add_argument('--features', action='store_true', help='Show feature importance')
    parser.add_argument('--info', action='store_true', help='Show model & board info')
    parser.add_argument('--export', action='store_true', help='Export results to CSV')
    parser.add_argument('--full-report', action='store_true', help='Run all analyses')
    parser.add_argument('-n', type=int, default=20, help='Samples per class for metrics/confusion')
    args = parser.parse_args()

    engine, data = load_engine()

    if args.full_report:
        run_model_info(engine)
        run_single_demo(engine, data)
        run_batch_test(engine, data, args.batch)
        run_confusion_matrix(engine, data, args.n)
        run_metrics(engine, data, args.n)
        run_feature_importance(data)
        run_attention_analysis(engine, data)
        run_benchmark(engine, data, 30)
        run_export(engine, data, args.n)
    elif args.bench > 0:
        run_benchmark(engine, data, args.bench)
    elif args.all:
        run_all_test(engine, data)
    elif args.confusion:
        run_confusion_matrix(engine, data, args.n)
    elif args.metrics:
        run_metrics(engine, data, args.n)
    elif args.attention:
        run_attention_analysis(engine, data)
    elif args.features:
        run_feature_importance(data)
    elif args.info:
        run_model_info(engine)
    elif args.export:
        run_export(engine, data, args.n)
    else:
        if not args.no_demo:
            run_single_demo(engine, data, args.cls)
        run_batch_test(engine, data, args.batch)

    print(f'\n{BOLD}{"═"*60}')
    print(f'  DONE')
    print(f'{"═"*60}{RST}')


if __name__ == '__main__':
    main()

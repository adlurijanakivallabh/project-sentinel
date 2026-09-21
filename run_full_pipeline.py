import sys
import time
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src import config
from src.data_preprocessing import load_and_preprocess
from src.sessionization import build_sequences_from_dataframe, save_sequences, load_sequences
from src.train import get_model, SequenceDataset
from src.losses import get_loss_function


def step_1_preprocess():
    print("\n" + "=" * 60)
    print("  STEP 1/6: DATA PREPROCESSING")
    print("=" * 60)
    
    csv_path = config.DATASETS_DIR / "cicids2017_preprocessed.csv"
    df = load_and_preprocess(save_path=csv_path)
    print(f"\n  [OK] Preprocessed data saved: {csv_path}")
    print(f"  [OK] Shape: {df.shape}")
    return df


def step_2_build_sequences(df):
    print("\n" + "=" * 60)
    print("  STEP 2/6: BUILDING SEQUENCES")
    print("=" * 60)
    
    print(f"  Temporal features: {'ENABLED (+5)' if config.TEMPORAL_FEATURES_ENABLED else 'DISABLED'}")
    print(f"  Window: {config.SEQUENCE_LENGTH} steps, stride: {config.SEQUENCE_STEP}")
    
    seqs, lbls, _ = build_sequences_from_dataframe(df)
    save_sequences(seqs, lbls, config.SEQUENCES_PATH)
    
    print(f"\n  [OK] Sequences: {seqs.shape[0]:,}")
    print(f"  [OK] Shape: {tuple(seqs.shape)}")
    print(f"  [OK] Features: {seqs.shape[2]} (25 base + 5 temporal)")
    
    for i, name in enumerate(config.get_class_names()):
        count = (lbls == i).sum().item()
        print(f"    {name}: {count:,} ({count/len(lbls)*100:.1f}%)")
    
    return seqs, lbls


def step_3_train(seqs, lbls):
    print("\n" + "=" * 60)
    print("  STEP 3/6: TRAINING CNN-LSTM")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    num_features = seqs.shape[2]
    model = get_model(config.MODEL_VARIANT, num_features=num_features)
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: {config.MODEL_VARIANT}")
    print(f"  Parameters: {param_count:,}")
    
    n = len(lbls)
    torch.manual_seed(config.RANDOM_STATE)
    indices = torch.randperm(n)
    split_train = int(n * (1 - config.TEST_SIZE - config.VAL_SIZE))
    split_val = int(n * (1 - config.TEST_SIZE))
    
    train_idx = indices[:split_train]
    val_idx = indices[split_train:split_val]
    test_idx = indices[split_val:]
    
    train_ds = SequenceDataset(seqs[train_idx], lbls[train_idx])
    val_ds = SequenceDataset(seqs[val_idx], lbls[val_idx])
    
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    criterion = get_loss_function(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n  Training for {config.NUM_EPOCHS} epochs...")
    print(f"  {'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Acc':>10} {'LR':>12}")
    print(f"  {'-'*56}")
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        running_loss = 0
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total * 100
        
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)
        
        val_acc = val_correct / val_total * 100
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_CHECKPOINT_PATH)
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or epoch == 1 or epoch == config.NUM_EPOCHS:
            print(f"  {epoch:>6} {train_loss:>12.4f} {train_acc:>9.2f}% {val_acc:>9.2f}% {lr:>12.6f}"
                  f"{'  *best*' if patience_counter == 0 else ''}")
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={config.EARLY_STOPPING_PATIENCE})")
            break
    
    print(f"\n  [OK] Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  [OK] Model saved to: {config.MODEL_CHECKPOINT_PATH}")
    
    torch.save({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx},
               config.SPLIT_INDICES_PATH)
    
    return model, test_idx


def step_4_evaluate(model, seqs, lbls, test_idx):
    print("\n" + "=" * 60)
    print("  STEP 4/6: EVALUATION")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(
        torch.load(config.MODEL_CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    
    test_seqs = seqs[test_idx]
    test_lbls = lbls[test_idx].numpy()
    
    all_preds = []
    all_probs = []
    latencies = []
    
    with torch.no_grad():
        for i in range(0, len(test_seqs), 256):
            batch = test_seqs[i:i+256].to(device)
            t0 = time.perf_counter()
            out = model(batch)
            t1 = time.perf_counter()
            
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            latencies.append((t1 - t0) / len(batch) * 1000)
    
    all_preds = np.array(all_preds)
    
    from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
    
    class_names = config.get_class_names()
    
    print("\n  Classification Report:")
    print("  " + "-" * 60)
    report = classification_report(test_lbls, all_preds, 
                                   target_names=class_names, digits=4)
    for line in report.split("\n"):
        print(f"  {line}")
    
    mcc = matthews_corrcoef(test_lbls, all_preds)
    print(f"\n  Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"  Mean inference latency: {np.mean(latencies):.3f} ms/sample")
    
    cm = confusion_matrix(test_lbls, all_preds)
    print("\n  Confusion Matrix:")
    header = "  " + f"{'':>20}" + "".join(f"{n[:8]:>10}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:>20}" + "".join(f"{v:>10}" for v in row))
    
    eval_path = config.RESULTS_DIR / "evaluation_results.txt"
    with open(eval_path, "w") as f:
        f.write("CNN-LSTM IPS — Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {config.MODEL_VARIANT}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Mean Latency: {np.mean(latencies):.3f} ms/sample\n\n")
        f.write(report)
    print(f"\n  [OK] Evaluation saved to: {eval_path}")
    
    return model


def step_5_fpga_model(seqs, lbls):
    print("\n" + "=" * 60)
    print("  STEP 5/6: FPGA MODEL (Transfer + Fine-tune)")
    print("=" * 60)
    
    from src.model_cnn_fpga import build_fpga_model, transfer_cnn_weights, fine_tune, print_model_summary
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = seqs.shape[2]
    
    fpga_model = build_fpga_model(num_features=num_features)
    print_model_summary(fpga_model)
    
    print("\n  Transferring CNN weights from CNN-LSTM...")
    fpga_model = transfer_cnn_weights(fpga_model)
    
    print(f"\n  Fine-tuning on {device} (10 epochs, CNN frozen)...")
    fpga_model = fine_tune(fpga_model, epochs=10, freeze_cnn=True)
    
    print(f"\n  [OK] FPGA model saved to: {config.FPGA_MODEL_PATH}")
    return fpga_model


def step_6_export():
    print("\n" + "=" * 60)
    print("  STEP 6/6: ONNX EXPORT + INT8 QUANTIZATION")
    print("=" * 60)
    
    from src.fpga_export import (
        load_trained_model, export_onnx, verify_onnx,
        quantize_dynamic, compare_accuracy, report_model_sizes,
        get_calibration_data, generate_export_report
    )
    
    print("\n  [6a] Exporting CNN-LSTM to ONNX...")
    model = load_trained_model()
    onnx_path = export_onnx(model)
    
    print("\n  [6b] Verifying ONNX output...")
    onnx_ok = verify_onnx(model, onnx_path)
    
    print("\n  [6c] Exporting FPGA model to ONNX...")
    from src.model_cnn_fpga import build_fpga_model
    
    if config.FPGA_MODEL_PATH.exists():
        fpga_model = build_fpga_model(num_features=len(config.FEATURE_COLUMNS) + 
                                       (5 if config.TEMPORAL_FEATURES_ENABLED else 0))
        fpga_model.load_state_dict(
            torch.load(config.FPGA_MODEL_PATH, map_location="cpu", weights_only=True)
        )
        fpga_model.eval()
        num_features = len(config.FEATURE_COLUMNS) + (5 if config.TEMPORAL_FEATURES_ENABLED else 0)
        export_onnx(fpga_model, output_path=config.ONNX_FPGA_MODEL_PATH,
                     num_features=num_features)
        print(f"  [OK] FPGA ONNX saved to: {config.ONNX_FPGA_MODEL_PATH}")
    
    print("\n  [6d] Applying INT8 quantization...")
    model_int8 = quantize_dynamic(model)
    torch.save(model_int8.state_dict(), config.QUANTIZED_MODEL_PATH)
    print(f"  [OK] INT8 model saved to: {config.QUANTIZED_MODEL_PATH}")
    
    print("\n  [6e] Comparing FP32 vs INT8...")
    test_seqs, test_lbls = get_calibration_data(n_samples=2000)
    acc_results = compare_accuracy(model, model_int8, test_seqs, test_lbls)
    size_results = report_model_sizes(config.MODEL_CHECKPOINT_PATH, model_int8)
    
    generate_export_report(acc_results, size_results, onnx_ok)
    
    print("\n  [OK] All FPGA export files ready!")


def main():
    start = time.time()
    
    print("=" * 60)
    print("  CNN-LSTM IPS — FULL PIPELINE RUNNER")
    print("=" * 60)
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Variant: {config.MODEL_VARIANT}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Classes: {config.NUM_CLASSES}")
    
    try:
        df = step_1_preprocess()
        
        seqs, lbls = step_2_build_sequences(df)
        del df
        
        model, test_idx = step_3_train(seqs, lbls)
        
        model = step_4_evaluate(model, seqs, lbls, test_idx)
        
        fpga_model = step_5_fpga_model(seqs, lbls)
        
        step_6_export()
        
    except Exception as e:
        print(f"\n  [FAIL] PIPELINE FAILED: {e}")
        traceback.print_exc()
        return
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 60)
    print("  [OK] PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"\n  Output files for FPGA:")
    print(f"    CNN-LSTM model  : {config.MODEL_CHECKPOINT_PATH}")
    print(f"    FPGA model      : {config.FPGA_MODEL_PATH}")
    print(f"    CNN-LSTM ONNX   : {config.ONNX_MODEL_PATH}")
    print(f"    FPGA ONNX       : {config.ONNX_FPGA_MODEL_PATH}")
    print(f"    INT8 quantized  : {config.QUANTIZED_MODEL_PATH}")
    print(f"    Export report   : {config.FPGA_DIR / 'export_report.txt'}")
    print(f"    Eval report     : {config.RESULTS_DIR / 'evaluation_results.txt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

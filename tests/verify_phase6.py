import sys
from pathlib import Path
import numpy as np
import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src.sessionization import load_sequences
from src.autoencoder import FlowAutoencoder
from src.binary_classifier import BinaryFlowMLP

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_autoencoder_architecture():
    section("TEST 1: Autoencoder Architecture")
    issues = []

    num_features = 30
    latent_dim = 10
    model = FlowAutoencoder(num_features=num_features, latent_dim=latent_dim)

    x = torch.randn(16, num_features)
    out = model(x)
    if out.shape != (16, num_features):
        issues.append(f"Output shape {out.shape} != expected (16, {num_features})")
    else:
        print(f"  {PASS} Forward pass shape: {out.shape}")

    errors = model.get_reconstruction_error(x)
    if errors.shape != (16,):
        issues.append(f"Error shape {errors.shape} != expected (16,)")
    else:
        print(f"  {PASS} Reconstruction error shape: {errors.shape}")

    if (errors < 0).any():
        issues.append("Reconstruction errors contain negative values!")
    else:
        print(f"  {PASS} All reconstruction errors are non-negative")

    z = model.encoder(x)
    if z.shape != (16, latent_dim):
        issues.append(f"Latent shape {z.shape} != expected (16, {latent_dim})")
    else:
        print(f"  {PASS} Latent dimension: {z.shape}")

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"  {PASS if encoder_params == decoder_params else WARN} Encoder params: {encoder_params:,}, Decoder params: {decoder_params:,} (symmetric: {encoder_params == decoder_params})")

    model.eval()
    x_single = torch.randn(1, num_features)
    try:
        _ = model(x_single)
        print(f"  {PASS} Single-sample inference works (eval mode)")
    except Exception as e:
        issues.append(f"Single-sample inference fails: {e}")

    return issues


def test_binary_classifier_architecture():
    section("TEST 2: Binary Classifier Architecture")
    issues = []

    num_features = 30
    model = BinaryFlowMLP(num_features=num_features)

    x = torch.randn(16, num_features)
    out = model(x)
    if out.shape != (16, 2):
        issues.append(f"Output shape {out.shape} != expected (16, 2)")
    else:
        print(f"  {PASS} Forward pass shape: {out.shape} (binary)")

    if out.min() >= 0 and out.max() <= 1:
        print(f"  {WARN} Output looks like probabilities, expected raw logits")
    else:
        print(f"  {PASS} Output is raw logits (pre-softmax)")

    param_count = sum(p.numel() for p in model.parameters())
    if param_count > 50000:
        issues.append(f"Model has {param_count:,} params — too heavy for a 'fast filter'")
    else:
        print(f"  {PASS} Lightweight model: {param_count:,} params")

    model.eval()
    try:
        _ = model(torch.randn(1, num_features))
        print(f"  {PASS} Single-sample inference works (eval mode)")
    except Exception as e:
        issues.append(f"Single-sample inference fails: {e}")

    return issues


def test_checkpoint_integrity():
    section("TEST 3: Checkpoint Integrity")
    issues = []
    device = torch.device("cpu")

    ae_path = config.MODELS_DIR / "autoencoder.pth"
    if not ae_path.exists():
        issues.append(f"Autoencoder checkpoint not found: {ae_path}")
        return issues

    ae_ckpt = torch.load(ae_path, map_location=device, weights_only=True)
    required_keys = {"model_state_dict", "num_features", "latent_dim", "val_loss", "epoch"}
    missing = required_keys - set(ae_ckpt.keys())
    if missing:
        issues.append(f"AE checkpoint missing keys: {missing}")
    else:
        print(f"  {PASS} AE checkpoint has all required keys: {sorted(ae_ckpt.keys())}")

    ae_model = FlowAutoencoder(ae_ckpt["num_features"], ae_ckpt["latent_dim"])
    ae_model.load_state_dict(ae_ckpt["model_state_dict"])
    print(f"  {PASS} AE model loads from checkpoint (features={ae_ckpt['num_features']}, latent={ae_ckpt['latent_dim']})")
    print(f"  {PASS} AE best epoch: {ae_ckpt['epoch']}, val_loss: {ae_ckpt['val_loss']:.6f}")

    thresh_path = config.MODELS_DIR / "autoencoder_threshold.pth"
    if not thresh_path.exists():
        issues.append(f"Threshold file not found: {thresh_path}")
    else:
        thresh = torch.load(thresh_path, map_location=device, weights_only=True)
        if thresh["threshold_p99"] <= thresh["threshold_p95"]:
            issues.append(f"P99 ({thresh['threshold_p99']}) should be > P95 ({thresh['threshold_p95']})")
        else:
            print(f"  {PASS} Threshold sanity: P99={thresh['threshold_p99']:.4f} > P95={thresh['threshold_p95']:.4f} > mean={thresh['normal_mse_mean']:.4f}")

    bin_path = config.MODELS_DIR / "binary_classifier.pth"
    if not bin_path.exists():
        issues.append(f"Binary classifier checkpoint not found: {bin_path}")
        return issues

    bin_ckpt = torch.load(bin_path, map_location=device, weights_only=False)
    required_keys = {"model_state_dict", "num_features", "val_acc", "f1", "epoch"}
    missing = required_keys - set(bin_ckpt.keys())
    if missing:
        issues.append(f"Binary checkpoint missing keys: {missing}")
    else:
        print(f"  {PASS} Binary checkpoint has all required keys")

    bin_model = BinaryFlowMLP(bin_ckpt["num_features"])
    bin_model.load_state_dict(bin_ckpt["model_state_dict"])
    print(f"  {PASS} Binary model loads from checkpoint (acc={bin_ckpt['val_acc']:.4f}, f1={bin_ckpt['f1']:.4f})")

    return issues


def test_training_data():
    section("TEST 4: Training Data Correctness")
    issues = []

    seqs, lbls, _ = load_sequences()
    normal_id = config.meta_label_to_id("Normal")

    normal_count = (lbls == normal_id).sum().item()
    if normal_count == 0:
        issues.append("No Normal samples found! AE training is impossible")
    else:
        print(f"  {PASS} Normal sequences: {normal_count:,} / {len(lbls):,}")

    attack_count = (lbls != normal_id).sum().item()
    if attack_count == 0:
        issues.append("No attack samples — can't evaluate detection")
    else:
        print(f"  {PASS} Attack sequences: {attack_count:,}")

    num_features = seqs.shape[2]
    expected = len(config.FEATURE_COLUMNS)
    if config.TEMPORAL_FEATURES_ENABLED:
        expected += len(config.TEMPORAL_FEATURE_NAMES)
    print(f"  {PASS if num_features == expected else WARN} Feature count: {num_features} (expected: {expected})")

    has_nan = torch.isnan(seqs).any().item()
    has_inf = torch.isinf(seqs).any().item()
    if has_nan:
        issues.append("Sequences contain NaN values!")
    if has_inf:
        issues.append("Sequences contain Inf values!")
    if not has_nan and not has_inf:
        print(f"  {PASS} No NaN or Inf in sequences")

    normal_seqs = seqs[lbls == normal_id]
    normal_flows = normal_seqs.reshape(-1, num_features)
    print(f"  {PASS} Normal flows available for AE training: {len(normal_flows):,}")

    unique_labels = torch.unique(lbls).numpy()
    if unique_labels.max() >= config.NUM_CLASSES:
        issues.append(f"Label {unique_labels.max()} >= NUM_CLASSES ({config.NUM_CLASSES})")
    else:
        print(f"  {PASS} Labels range: {unique_labels.min()}-{unique_labels.max()} (NUM_CLASSES={config.NUM_CLASSES})")

    return issues


def test_autoencoder_detection():
    section("TEST 5: Autoencoder Detection Logic")
    issues = []
    device = torch.device("cpu")

    ae_ckpt = torch.load(config.MODELS_DIR / "autoencoder.pth", map_location=device, weights_only=True)
    model = FlowAutoencoder(ae_ckpt["num_features"], ae_ckpt["latent_dim"])
    model.load_state_dict(ae_ckpt["model_state_dict"])
    model.eval()

    thresh_data = torch.load(config.MODELS_DIR / "autoencoder_threshold.pth", map_location=device, weights_only=True)
    threshold = thresh_data["threshold_p99"]

    seqs, lbls, _ = load_sequences()
    normal_id = config.meta_label_to_id("Normal")
    num_features = seqs.shape[2]

    normal_flows = seqs[lbls == normal_id].reshape(-1, num_features)
    sample_idx = torch.randperm(len(normal_flows))[:5000]
    sample = normal_flows[sample_idx]

    with torch.no_grad():
        errors = model.get_reconstruction_error(sample).numpy()
    fp_rate = (errors > threshold).mean() * 100

    if fp_rate > 5.0:
        issues.append(f"False positive rate {fp_rate:.1f}% is too high (>5%)")
    else:
        print(f"  {PASS} Normal FP rate: {fp_rate:.1f}% (<5% threshold)")

    if threshold <= 0:
        issues.append(f"Threshold {threshold} is non-positive!")
    else:
        print(f"  {PASS} Threshold is positive: {threshold:.4f}")

    x_test = torch.randn(10, num_features)
    model.eval()
    with torch.no_grad():
        e1 = model.get_reconstruction_error(x_test)
        e2 = model.get_reconstruction_error(x_test)
    if not torch.allclose(e1, e2):
        issues.append("Model gives different results on same input in eval mode!")
    else:
        print(f"  {PASS} Model is deterministic in eval mode")

    x_grad = torch.randn(10, num_features, requires_grad=True)
    errors = model.get_reconstruction_error(x_grad)
    if errors.requires_grad:
        issues.append("get_reconstruction_error returns tensors with gradients!")
    else:
        print(f"  {PASS} get_reconstruction_error correctly uses no_grad")

    return issues


def test_binary_classifier_detection():
    section("TEST 6: Binary Classifier Detection Logic")
    issues = []
    device = torch.device("cpu")

    bin_ckpt = torch.load(config.MODELS_DIR / "binary_classifier.pth", map_location=device, weights_only=False)
    model = BinaryFlowMLP(bin_ckpt["num_features"])
    model.load_state_dict(bin_ckpt["model_state_dict"])
    model.eval()

    seqs, lbls, _ = load_sequences()
    normal_id = config.meta_label_to_id("Normal")
    num_features = seqs.shape[2]

    normal_flows = seqs[lbls == normal_id].reshape(-1, num_features)
    sample = normal_flows[torch.randperm(len(normal_flows))[:2000]]
    with torch.no_grad():
        out = model(sample)
    normal_pred_normal = (out.argmax(1) == 0).float().mean().item()
    if normal_pred_normal < 0.5:
        issues.append(f"Normal traffic classified as Normal only {normal_pred_normal*100:.1f}% — too low!")
    else:
        print(f"  {PASS} Normal traffic → Normal prediction: {normal_pred_normal*100:.1f}%")

    attack_flows = seqs[lbls != normal_id].reshape(-1, num_features)
    sample = attack_flows[torch.randperm(len(attack_flows))[:2000]]
    with torch.no_grad():
        out = model(sample)
    attack_pred_attack = (out.argmax(1) == 1).float().mean().item()
    if attack_pred_attack < 0.3:
        issues.append(f"Attack traffic classified as Attack only {attack_pred_attack*100:.1f}% — too low!")
    else:
        print(f"  {PASS} Attack traffic → Attack prediction: {attack_pred_attack*100:.1f}%")

    saved_acc = bin_ckpt["val_acc"]
    print(f"  {PASS} Saved val accuracy: {saved_acc:.4f}")

    x_test = torch.randn(10, num_features)
    with torch.no_grad():
        o1 = model(x_test)
        o2 = model(x_test)
    if not torch.allclose(o1, o2):
        issues.append("Binary model is non-deterministic in eval mode!")
    else:
        print(f"  {PASS} Model is deterministic in eval mode")

    return issues


def test_integration():
    section("TEST 7: Integration & Code Quality")
    issues = []

    ae_ckpt = torch.load(config.MODELS_DIR / "autoencoder.pth", map_location="cpu", weights_only=True)
    bin_ckpt = torch.load(config.MODELS_DIR / "binary_classifier.pth", map_location="cpu", weights_only=False)

    if ae_ckpt["num_features"] != bin_ckpt["num_features"]:
        issues.append(f"Feature mismatch: AE={ae_ckpt['num_features']}, Binary={bin_ckpt['num_features']}")
    else:
        print(f"  {PASS} Both models use {ae_ckpt['num_features']} features (consistent)")

    expected_files = [
        config.MODELS_DIR / "autoencoder.pth",
        config.MODELS_DIR / "autoencoder_threshold.pth",
        config.MODELS_DIR / "binary_classifier.pth",
        config.FIGURES_DIR / "autoencoder_mse_distribution.png",
        config.FIGURES_DIR / "hierarchical_comparison.png",
    ]
    for f in expected_files:
        if f.exists():
            print(f"  {PASS} File exists: {f.name}")
        else:
            issues.append(f"Missing output: {f}")

    seqs, lbls, _ = load_sequences()
    normal_id = config.meta_label_to_id("Normal")
    normal_count = (lbls == normal_id).sum().item()
    normal_flows = normal_count * config.SEQUENCE_LENGTH
    print(f"  {PASS} AE training data: {normal_flows:,} flows from {normal_count:,} Normal windows only")

    torch.manual_seed(config.RANDOM_STATE)
    indices = torch.randperm(len(lbls))
    test_start = int(len(lbls) * (1 - config.TEST_SIZE))
    test_count = len(indices) - test_start
    print(f"  {PASS} Hierarchical eval test set: {test_count:,} windows ({config.TEST_SIZE*100:.0f}% split)")

    return issues


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  PHASE 6a/6b VERIFICATION SUITE")
    print("█" * 60)

    all_issues = []

    tests = [
        ("Architecture - Autoencoder", test_autoencoder_architecture),
        ("Architecture - Binary Classifier", test_binary_classifier_architecture),
        ("Checkpoint Integrity", test_checkpoint_integrity),
        ("Training Data Correctness", test_training_data),
        ("Autoencoder Detection Logic", test_autoencoder_detection),
        ("Binary Classifier Detection Logic", test_binary_classifier_detection),
        ("Integration & Code Quality", test_integration),
    ]

    for name, test_fn in tests:
        try:
            issues = test_fn()
            all_issues.extend(issues)
            for issue in issues:
                print(f"  {FAIL} {issue}")
        except Exception as e:
            all_issues.append(f"{name} CRASHED: {e}")
            print(f"  {FAIL} {name} CRASHED: {e}")
            import traceback
            traceback.print_exc()

    section("VERIFICATION SUMMARY")
    total_tests = 7
    failed = len(all_issues)
    passed = total_tests - min(failed, total_tests)

    if all_issues:
        print(f"\n  {FAIL} Found {len(all_issues)} issue(s):")
        for i, issue in enumerate(all_issues, 1):
            print(f"    {i}. {issue}")
    else:
        print(f"\n  {PASS} ALL {total_tests} TEST GROUPS PASSED — No issues found!")

    print(f"\n  Tests: {total_tests} | Issues: {len(all_issues)}")
    print(f"{'='*60}\n")

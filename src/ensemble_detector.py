import sys
from pathlib import Path
import numpy as np
import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import config
from src import clean_state_dict
from src.sessionization import load_sequences
from src.train import get_model
from src.split_utils import block_split
from src.autoencoder_v2 import WindowLSTMAutoencoder, WindowNormalizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("  E: ENSEMBLE — AUTOENCODER V2 + CNN-LSTM-ATTENTION")
    print("=" * 60)

    seqs, lbls, _ = load_sequences()
    _, _, test_idx = block_split(len(lbls))
    test_seqs, test_lbls = seqs[test_idx], lbls[test_idx]
    names = config.get_class_names()
    test_np = test_lbls.numpy()

    ckpt = torch.load(config.MODEL_CHECKPOINT_PATH, map_location=device, weights_only=False)
    cnn_model = get_model("cnn_lstm_attention", num_features=30)
    cnn_model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    cnn_model.to(device).eval()

    cnn_preds = []
    with torch.no_grad():
        for i in range(0, len(test_seqs), 256):
            batch = test_seqs[i:i+256].to(device)
            preds = cnn_model(batch).argmax(1).cpu().tolist()
            cnn_preds.extend(preds)
    cnn_preds = np.array(cnn_preds)

    ae_path = config.MODELS_DIR / "autoencoder_v2.pth"
    ae_ckpt = torch.load(ae_path, map_location=device, weights_only=False)
    ae_model = WindowLSTMAutoencoder(
        num_features=ae_ckpt["num_features"],
        seq_len=ae_ckpt["seq_len"],
        hidden_dim=ae_ckpt["hidden_dim"],
        latent_dim=ae_ckpt["latent_dim"],
    )
    ae_model.load_state_dict(clean_state_dict(ae_ckpt["model_state_dict"]))
    ae_model.to(device).eval()

    normalizer = WindowNormalizer.load(config.MODELS_DIR / "autoencoder_v2_normalizer.pkl")
    thresh_data = torch.load(
        config.MODELS_DIR / "autoencoder_v2_threshold.pth",
        map_location="cpu", weights_only=True,
    )

    ae_errors = []
    with torch.no_grad():
        for i in range(0, len(test_seqs), 256):
            batch = normalizer.transform(test_seqs[i:i+256]).to(device)
            errors = ae_model.get_reconstruction_error(batch)
            ae_errors.extend(errors.cpu().numpy())
    ae_errors = np.array(ae_errors)

    print(f"\n  Test windows: {len(test_seqs):,}")
    print(f"  CNN-LSTM accuracy: {(cnn_preds == test_np).mean()*100:.2f}%")

    normal_id = config.meta_label_to_id("Normal")

    for tkey in ["threshold_p99", "threshold_p97", "threshold_p95"]:
        threshold = thresh_data[tkey]
        ae_flagged = ae_errors > threshold

        print(f"\n  --- Ensemble with AE {tkey} = {threshold:.4f} ---")

        ensemble_preds = cnn_preds.copy()
        overrides = 0
        correct_overrides = 0
        incorrect_overrides = 0

        for i in range(len(ensemble_preds)):
            if ensemble_preds[i] == normal_id and ae_flagged[i]:
                overrides += 1
                if test_np[i] != normal_id:
                    correct_overrides += 1
                else:
                    incorrect_overrides += 1

        is_attack = test_np != normal_id
        cnn_detects_attack = cnn_preds[is_attack] != normal_id
        ae_detects_from_cnn_miss = ae_flagged[is_attack] & (cnn_preds[is_attack] == normal_id)

        cnn_attack_detection = cnn_detects_attack.sum()
        ensemble_attack_detection = cnn_attack_detection + ae_detects_from_cnn_miss.sum()
        total_attacks = is_attack.sum()

        is_normal = test_np == normal_id
        cnn_fp = (cnn_preds[is_normal] != normal_id).sum()
        ae_extra_fp = (ae_flagged[is_normal] & (cnn_preds[is_normal] == normal_id)).sum()
        ensemble_fp = cnn_fp + ae_extra_fp

        print(f"    Overrides (CNN=Normal, AE=Anomaly): {overrides}")
        print(f"      Correct (true attack): {correct_overrides}")
        print(f"      Incorrect (true normal): {incorrect_overrides}")
        print(f"    CNN-LSTM attack detection: {cnn_attack_detection}/{total_attacks} "
              f"({cnn_attack_detection/total_attacks*100:.1f}%)")
        print(f"    Ensemble attack detection: {ensemble_attack_detection}/{total_attacks} "
              f"({ensemble_attack_detection/total_attacks*100:.1f}%)")
        print(f"    Improvement: +{(ensemble_attack_detection-cnn_attack_detection)/total_attacks*100:.2f}%")
        print(f"    CNN FP: {cnn_fp}/{is_normal.sum()} ({cnn_fp/is_normal.sum()*100:.2f}%)")
        print(f"    Ensemble FP: {ensemble_fp}/{is_normal.sum()} ({ensemble_fp/is_normal.sum()*100:.2f}%)")

        print(f"\n    Per-class attack detection:")
        for cls_id in range(1, 6):
            cls_mask = test_np == cls_id
            if cls_mask.sum() == 0:
                continue
            cnn_det = (cnn_preds[cls_mask] != normal_id).sum()
            ae_extra = (ae_flagged[cls_mask] & (cnn_preds[cls_mask] == normal_id)).sum()
            total_cls = cls_mask.sum()
            print(f"      {names[cls_id]:25s}: CNN={cnn_det/total_cls*100:.1f}%, "
                  f"Ensemble={((cnn_det+ae_extra)/total_cls)*100:.1f}% "
                  f"(+{ae_extra} windows)")

    print(f"\n{'=' * 60}")
    print("  ENSEMBLE ANALYSIS COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

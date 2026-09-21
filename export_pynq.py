import torch
import sys
import os
import numpy as np

sys.path.insert(0, '.')
from src import config, clean_state_dict
from src.model_v2 import build_model_v2

ckpt = torch.load(config.MODELS_DIR / 'best_model_v2_swa.pth', map_location='cpu', weights_only=False)
model = build_model_v2(num_features=ckpt['num_features'], num_classes=ckpt.get('num_classes', 8))
model.load_state_dict(clean_state_dict(ckpt['model_state_dict']))
model.eval()
print("Model loaded:", ckpt['num_features'], "features,", ckpt.get('num_classes', 8), "classes")

dummy = torch.randn(1, 20, 30)
torch.onnx.export(model, dummy, 'sentinel_v2.onnx',
    export_params=True, opset_version=13, dynamo=False,
    do_constant_folding=True,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})

np.savez('zscore_stats.npz',
    mean=ckpt['zscore_mean'].cpu().numpy(),
    std=ckpt['zscore_std'].cpu().numpy())

onnx_size = os.path.getsize('sentinel_v2.onnx') / 1024
zscore_size = os.path.getsize('zscore_stats.npz') / 1024
print("ONNX exported: sentinel_v2.onnx (%d KB)" % int(onnx_size))
print("Z-score stats: zscore_stats.npz (%.1f KB)" % zscore_size)

try:
    import onnxruntime as ort
    sess = ort.InferenceSession('sentinel_v2.onnx')
    pt_out = model(dummy).detach().numpy()
    onnx_out = sess.run(None, {'input': dummy.numpy()})[0]
    diff = abs(pt_out - onnx_out).max()
    status = "PASS" if diff < 1e-4 else "FAIL"
    print("Verification: max diff = %.8f (%s)" % (diff, status))
except ImportError:
    print("onnxruntime not installed, skipping verification")

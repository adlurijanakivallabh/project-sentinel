import torch
from src import config
from src.model_cnn_fpga import build_fpga_model

print("Loading PyTorch FPGA model...")
num_features = len(config.FEATURE_COLUMNS)
if getattr(config, 'TEMPORAL_FEATURES_ENABLED', False):
    num_features += 5

model = build_fpga_model(num_features=num_features)
model.load_state_dict(torch.load("results/fpga/cnn_fpga.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, config.SEQUENCE_LENGTH, num_features)

print("Exporting to static ONNX...")
out_path = config.FPGA_DIR / "cnn_fpga_static.onnx"
torch.onnx.export(
    model,
    dummy_input,
    str(out_path),
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)

print(f"Done! Saved to {out_path}")

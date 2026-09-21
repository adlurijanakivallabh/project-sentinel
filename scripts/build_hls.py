import os
import traceback
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import hls4ml
from tensorflow import keras

def main():
    print("=" * 60)
    print("  CNN FPGA SYNTHESIS (hls4ml - Keras Backend)")
    print("=" * 60)

    h5_path = Path("results/fpga/cnn_fpga.h5")
    if not h5_path.exists():
        print(f"[FAIL] Keras model weights missing: {h5_path}")
        print("Run 'python pytorch_to_keras.py' first.")
        return

    print(f"[1/4] Loading Keras model from {h5_path}...")
    model = keras.models.load_model(str(h5_path), compile=False)

    print("[2/4] Configuring HLS for Xilinx Zynq-7020...")
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    
    hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
    hls_config['Model']['Strategy'] = 'Latency'
    hls_config['Model']['ReuseFactor'] = 1

    import pprint
    print("\n[CONFIG]")
    pprint.pprint(hls_config)
    print("\n")

    out_dir = Path("fpga_hls_project")

    print(f"[3/4] Generating C++ HLS project at {out_dir}...")
    try:
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=hls_config,
            output_dir=str(out_dir),
            part='xc7z020clg400-1',
            io_type='io_stream',
            clock_period=5
        )
        hls_model.write()
        print(f"  -> C++ Source files successfully generated in {out_dir}/")
    except Exception as e:
        print("[FAIL] hls4ml Keras conversion failed.")
        traceback.print_exc()
        return

    print("[4/4] Attempting Vivado Synthesis Build...")
    print("  (This requires 'vivado_hls' or 'vitis_hls' to be installed and in your PATH)")
    try:
        hls_model.build(csim=False, synth=True, vsynth=True)
        print("\n[OK] Synthesis complete. IP block generated successfully!")
    except Exception as e:
        print("\n[INFO] Synthesis step skipped. You can manually run Vivado synthesis on the generated C++ files.")

if __name__ == "__main__":
    main()

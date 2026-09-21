import argparse
import numpy as np

def deploy_zynq_ips(bitstream_path: str, input_features: np.ndarray):
    try:
        from pynq import Overlay
        from pynq import allocate
    except ImportError:
        print("[WARNING] PYNQ is not installed. Ensure you are running this on your Zynq-7020 board.")
        print("[WARNING] Exiting simulated PYNQ deployment skeleton.")
        return

    print(f"Loading bitstream: {bitstream_path}")
    overlay = Overlay(bitstream_path)

    dma = overlay.axi_dma_0

    num_features = 30
    num_classes = 5

    print("Allocating continuous memory buffers...")
    in_buffer = allocate(shape=(num_features,), dtype=np.float32)
    out_buffer = allocate(shape=(num_classes,), dtype=np.float32)

    np.copyto(in_buffer, input_features)

    import time
    print("Sending AXI stream to Neural Network IP...")
    start_time = time.perf_counter()

    dma.sendchannel.transfer(in_buffer)
    dma.recvchannel.transfer(out_buffer)
    
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    print(f"[OK] Inference complete in {duration_ms:.4f} ms")

    classes = ['Normal', 'DDoS/DoS', 'PortScan', 'Web/SQLi', 'Malware']
    predicted_idx = np.argmax(out_buffer)
    predicted_class = classes[predicted_idx]
    
    print("\n[RESULT]")
    print(f"Predicted Class: {predicted_class}")
    print(f"Logits: {out_buffer}")
    
    return predicted_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PYNQ Inference for CNN-LSTM IPS.")
    parser.add_argument("--bitstream", type=str, default="cnn_fpga_v1.bit", help="Path to your generated Vivado Bitstream")
    args = parser.parse_args()
    
    dummy_input = np.random.rand(30).astype(np.float32)
    
    deploy_zynq_ips(args.bitstream, dummy_input)

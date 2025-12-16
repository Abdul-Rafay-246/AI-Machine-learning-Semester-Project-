"""
Quick GPU check for PyTorch/Ultralytics.

Usage:
    python check_gpu.py
"""
import torch


def main():
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if not available:
        return

    count = torch.cuda.device_count()
    print(f"Visible CUDA devices: {count}")
    for idx in range(count):
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024**3)
        print(f"[{idx}] {name} | {total_gb:.1f} GB")

    # simple sanity tensor
    try:
        x = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
        print("Tensor on GPU:", x)
    except Exception as e:
        print("Failed to move tensor to GPU:", e)


if __name__ == "__main__":
    main()

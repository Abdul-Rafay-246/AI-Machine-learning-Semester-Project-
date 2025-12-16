"""
Train YOLOv8 on the augmented shapes dataset.

Requires:
    pip install ultralytics

Usage:
    python train_shapes.py
"""
from pathlib import Path
from ultralytics import YOLO


def main():
    script_dir = Path(__file__).resolve().parent          # .../myenv
    # dataset lives inside myenv/shapes_augmented (keep everything self-contained)
    data_yaml = script_dir / "shapes_augmented" / "data.yaml"
    output_dir = script_dir / "runs_shapes"               # keep outputs inside myenv

    # use a lighter model to fit on 6 GB GPU; switch to yolov8x.pt if you have more memory
    model = YOLO("yolov8s.pt")

    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=512,           # smaller image size to ease VRAM/compute
        batch=4,             # lighter batch for 6 GB GPU
        workers=2,           # fewer workers helps on Windows/CPU-bound aug
        auto_augment=None,   # disable heavy randaugment
        cache="ram",         # tiny dataset fits in RAM; speeds first epoch
        project=str(output_dir),
        name="yolov8s_shapes_aug",
        pretrained=True,
        device=0,  # force GPU 0;
    )


if __name__ == "__main__":
    main()

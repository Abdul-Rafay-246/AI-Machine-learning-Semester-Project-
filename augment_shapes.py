"""
Augment the Roboflow shapes dataset (YOLOv8 format) and write originals + augmented
samples into a new combined dataset folder.
"""
"""
Dataset download Command

pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="sB3LfHVJIkk2PlH2y6JP")
project = rf.workspace("work-bzti1").project("shape-detection-n1tzt")
version = project.version(6)
dataset = version.download("yolov8")

Dataset download Link
https://universe.roboflow.com/ds/gPx5AgnLS9?key=fMDjMEAfF1
"""              

import os
import shutil
from pathlib import Path

import cv2
import albumentations as A


# Configuration
NUM_AUG_PER_IMAGE = 3
SPLITS_TO_AUGMENT = ["train"]  # typically augment only training data


def load_labels(label_path: Path):
    """Load YOLO format labels. Returns list of (class_id, x, y, w, h)."""
    boxes = []
    if not label_path.exists():
        return boxes
    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            boxes.append((int(cls), float(x), float(y), float(w), float(h)))
    return boxes


def save_labels(label_path: Path, boxes):
    """Save YOLO format labels."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w") as f:
        for cls, x, y, w, h in boxes:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def get_transform():
    """Albumentations pipeline for bounding boxes in YOLO format."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-10, 10), p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2),
    )


def copy_original(image_path: Path, label_path: Path, out_images: Path, out_labels: Path):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, out_images / image_path.name)
    if label_path.exists():
        shutil.copy2(label_path, out_labels / label_path.name)


def augment_split(split_dir: Path, out_split_dir: Path, transform):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    out_images = out_split_dir / "images"
    out_labels = out_split_dir / "labels"

    for image_path in images_dir.glob("*"):
        if not image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"

        # copy original
        copy_original(image_path, label_path, out_images, out_labels)

        # load data
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        boxes = load_labels(label_path)
        class_labels = [cls for cls, _, _, _, _ in boxes]
        bbox_coords = [(x, y, w, h) for _, x, y, w, h in boxes]

        for idx in range(NUM_AUG_PER_IMAGE):
            augmented = transform(image=image, bboxes=bbox_coords, class_labels=class_labels)
            aug_img = augmented["image"]
            aug_boxes = [
                (cls, *bbox) for cls, bbox in zip(augmented["class_labels"], augmented["bboxes"])
            ]

            aug_name = f"{image_path.stem}_aug{idx}"
            aug_img_path = out_images / f"{aug_name}{image_path.suffix}"
            aug_label_path = out_labels / f"{aug_name}.txt"

            cv2.imwrite(str(aug_img_path), aug_img)
            save_labels(aug_label_path, aug_boxes)


def main():
    project_root = Path(__file__).resolve().parent.parent
    src_root = project_root / "shapes"
    dest_root = project_root / "shapes_augmented"

    transform = get_transform()

    for split in SPLITS_TO_AUGMENT:
        split_dir = src_root / split
        out_split_dir = dest_root / split
        if not split_dir.exists():
            print(f"Skipping missing split: {split_dir}")
            continue
        augment_split(split_dir, out_split_dir, transform)

    # copy untouched splits (e.g., val/test) if not augmented
    for split in ["valid", "test"]:
        if split in SPLITS_TO_AUGMENT:
            continue
        split_dir = src_root / split
        out_split_dir = dest_root / split
        if split_dir.exists():
            shutil.copytree(split_dir, out_split_dir, dirs_exist_ok=True)

    # copy data.yaml and adjust paths relative to new root
    yaml_src = src_root / "data.yaml"
    yaml_dest = dest_root / "data.yaml"
    if yaml_src.exists():
        text = yaml_src.read_text()
        # replace relative paths to point to new folders
        text = text.replace("../train/images", "train/images")
        text = text.replace("../valid/images", "valid/images")
        text = text.replace("../test/images", "test/images")
        yaml_dest.write_text(text)

    print(f"Augmented dataset written to: {dest_root}")


if __name__ == "__main__":
    main()

"""
Summarize training metrics and basic label stats using pandas and NumPy, plus simple plots.

Usage:
    python summarize_results.py

Dependencies:
    pip install pandas numpy matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(run_dir: Path) -> pd.DataFrame:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found at {results_csv}")
    return pd.read_csv(results_csv)


def summarize_training(df: pd.DataFrame) -> None:
    best_idx = df["metrics/mAP50(B)"].idxmax()
    best = df.loc[best_idx]
    final = df.iloc[-1]

    print("\n=== Training summary (easy to read) ===")
    print(f"- Total epochs: {len(df)}")
    print(f"- Best epoch (highest mAP50): {int(best['epoch'])}")
    print(
        f"  mAP50={best['metrics/mAP50(B)']:.3f} (overall detection quality at IoU 0.50)\n"
        f"  mAP50-95={best['metrics/mAP50-95(B)']:.3f} (harder IoU range 0.50-0.95)\n"
        f"  Precision={best['metrics/precision(B)']:.3f} (few false positives)\n"
        f"  Recall={best['metrics/recall(B)']:.3f} (few missed objects)"
    )
    print(
        f"- Final epoch {int(final['epoch'])}: "
        f"mAP50={final['metrics/mAP50(B)']:.3f}, "
        f"mAP50-95={final['metrics/mAP50-95(B)']:.3f}"
    )
    print(
        f"  Training losses (lower is better): "
        f"box={final['train/box_loss']:.3f}, "
        f"cls={final['train/cls_loss']:.3f}, "
        f"dfl={final['train/dfl_loss']:.3f}"
    )


def load_label_arrays(labels_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return class ids and areas (w*h) from YOLO txt labels."""
    class_ids = []
    areas = []
    for txt in labels_dir.glob("*.txt"):
        lines = txt.read_text().strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, _, _, w, h = map(float, parts)
            class_ids.append(int(cls))
            areas.append(w * h)
    return np.array(class_ids), np.array(areas)


def summarize_labels(dataset_root: Path) -> None:
    labels_dir = dataset_root / "train" / "labels"
    class_names = ["circle", "rectangle", "square", "triangle"]  # from data.yaml
    if not labels_dir.exists():
        print(f"\nLabel summary skipped: missing {labels_dir}")
        return
    class_ids, areas = load_label_arrays(labels_dir)
    if class_ids.size == 0:
        print(f"\nLabel summary skipped: no labels found in {labels_dir}")
        return

    print("\n=== Label summary (train split) ===")
    counts = pd.Series(class_ids).value_counts().sort_index()
    print("Class counts:")
    for cls_id, count in counts.items():
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        print(f"  class {cls_id} ({name}): {count}")

    print(
        f"Box area stats (normalized units): "
        f"min={areas.min():.4f}, median={np.median(areas):.4f}, max={areas.max():.4f}"
    )


def plot_curves(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
    axes[0].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    axes[0].set_title("Validation mAP vs epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train/box_loss"], label="box_loss")
    axes[1].plot(df["epoch"], df["train/cls_loss"], label="cls_loss")
    axes[1].plot(df["epoch"], df["train/dfl_loss"], label="dfl_loss")
    axes[1].set_title("Training losses vs epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    out_file = out_dir / "summary_curves.png"
    fig.savefig(out_file, dpi=200)
    print(f"\nSaved plots to {out_file}")
    try:
        # Show interactively; keep blocking so the window actually appears
        plt.show(block=True)
    except Exception as e:
        print(f"Could not open interactive window: {e}")
    finally:
        plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent
    run_dir = script_dir / "runs_shapes" / "yolov8s_shapes_aug3"
    dataset_root = script_dir / "shapes_augmented"

    df = load_results(run_dir)
    summarize_training(df)
    summarize_labels(dataset_root)
    plot_curves(df, run_dir)


if __name__ == "__main__":
    main()

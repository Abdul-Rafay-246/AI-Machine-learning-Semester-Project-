"""
Run the trained shapes model on your webcam for real-time detection.

Usage:
    python run_shapes_camera.py
Press 'q' to quit the window.
"""
from pathlib import Path

import cv2
from ultralytics import YOLO


def open_camera():
    """Try common Windows backends to open webcam."""
    attempts = [
        (0, None),
        (0, cv2.CAP_DSHOW),
        (0, cv2.CAP_MSMF),
    ]
    for idx, backend in attempts:
        cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap
        cap.release()
    raise RuntimeError("Could not open webcam (tried device 0 with default/DShow/MSMF)")


def check_highgui_or_die():
    """Ensure OpenCV has GUI support; otherwise give a clear error."""
    try:
        cv2.namedWindow("highgui_test")
        cv2.destroyWindow("highgui_test")
    except cv2.error as e:
        msg = (
            "Your OpenCV build lacks GUI (highgui) support; install the full wheel:\n"
            "  .\\Scripts\\python -m pip uninstall -y opencv-python-headless\n"
            "  .\\Scripts\\python -m pip install --upgrade opencv-python\n"
            f"Original error: {e}"
        )
        raise RuntimeError(msg)


def main():
    script_dir = Path(__file__).resolve().parent
    weights_path = script_dir / "runs_shapes" / "yolov8s_shapes_aug3" / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Trained weights not found at {weights_path}")

    model = YOLO(str(weights_path))

    check_highgui_or_die()
    cap = open_camera()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam; exiting.")
                break

            results = model.predict(frame, device=0, conf=0.5, verbose=False)
            annotated = results[0].plot()

            cv2.imshow("Shapes detection (press q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # If running with a headless OpenCV build, skip closing windows gracefully
            pass


if __name__ == "__main__":
    main()

import cv2
from ultralytics import YOLO


def main():
    model = YOLO("yolov8x.pt")  # use local weights file

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("YOLOv8 Webcam (press q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

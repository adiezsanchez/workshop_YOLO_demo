"""YOLO26 segmentation on the webcam. Press q to close the window."""

from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def open_camera():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")
    return camera


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    weights = Path(__file__).resolve().parent.parent / "models" / "yolo26n-seg.pt"
    model = YOLO(weights)
    camera = open_camera()
    try:
        while True:
            ok, frame = camera.read()
            if ok:
                annotated = model.predict(frame, device=device, verbose=False)[0].plot()
                cv2.imshow("YOLO segmentation", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

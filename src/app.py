import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # YOLO on the webcam

    Each frame from the webcam is run through a **YOLO26** model on the GPU.

    - **Object detection** draws a box around every object, with its class and confidence.
    - **Segmentation** does the same, and also fills in the pixels that belong to that object.

    Switch tasks with the buttons below. The weights download the first time this notebook runs.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    if device == 0:
        torch.backends.cudnn.benchmark = True

    weights = Path(__file__).resolve().parent.parent / "models"
    weights.mkdir(exist_ok=True)
    models = {
        "Object detection": YOLO(weights / "yolo26n.pt"),
        "Segmentation": YOLO(weights / "yolo26n-seg.pt"),
    }

    # First call pays for weight loading and CUDA warmup, so the live view stays smooth.
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for model in models.values():
        model.predict(blank, device=device, verbose=False)

    def _open_camera():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open the webcam.")
        return cap

    class _Webcam:
        # A failed read leaves some Windows cameras stuck, so reopen and try once more.
        def __init__(self):
            self._cap = _open_camera()

        def read(self):
            ok, frame = self._cap.read()
            if ok:
                return frame
            self._cap.release()
            self._cap = _open_camera()
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")
            return frame

    camera = _Webcam()
    return Image, camera, cv2, device, device_name, models


@app.cell
def _(mo):
    task = mo.ui.radio(
        options=["Object detection", "Segmentation"],
        value="Object detection",
        inline=True,
        label="Task",
    )
    refresh = mo.ui.refresh(
        options=["100ms", "200ms", "500ms", "1s"],
        default_interval="100ms",
    )
    mo.hstack([task, refresh], align="end", gap=2)
    return refresh, task


@app.cell
def _(Image, camera, cv2, device, device_name, mo, models, refresh, task):
    _ = refresh.value

    try:
        frame = camera.read()
    except RuntimeError as error:
        mo.stop(True, mo.md(f"**{error}**"))

    result = models[task.value].predict(frame, device=device, verbose=False)[0]
    rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

    found = "nothing detected"
    if len(result.boxes):
        counts = {}
        for cls in result.boxes.cls:
            name = result.names[int(cls)]
            counts[name] = counts.get(name, 0) + 1
        found = ", ".join(f"{name} × {count}" for name, count in counts.items())

    caption = f"{task.value} · {device_name} · {result.speed['inference']:.0f} ms · {found}"
    mo.image(Image.fromarray(rgb), caption=caption, width="100%")
    return


if __name__ == "__main__":
    app.run()

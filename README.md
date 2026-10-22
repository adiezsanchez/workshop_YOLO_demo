# YOLO webcam demo

A [marimo](https://marimo.io) notebook that runs [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) on a live webcam feed. Each frame is drawn with what the model found, so you can see how YOLO works.

The notebook loads two pretrained models:

- **Object detection** (`yolo26n.pt`) draws a box around each object, with its class and confidence.
- **Segmentation** (`yolo26n-seg.pt`) does the same, and also fills in the pixels that belong to that object.

A radio button switches between the two. Inference uses the GPU through PyTorch when CUDA is available, and falls back to the CPU otherwise. The caption under the image shows the device, the inference time, and the objects in the current frame.

## Prerequisites

- [pixi](https://github.com/prefix-dev/pixi)
- A webcam
- An NVIDIA GPU with a driver that supports CUDA 12.8 (this environment installs the CUDA build of PyTorch)

## Getting started

```bash
git clone https://github.com/adiezsanchez/workshop_YOLO_demo
cd workshop_YOLO_demo
pixi run demo
```

`pixi run demo` creates the environment on first use and opens `src/app.py` in the marimo editor. Run the notebook from the editor. The first run downloads the weights into `models/`.

## Using the demo

1. Allow the notebook to open the webcam when it starts.
2. Choose **Object detection** or **Segmentation**. The view updates as soon as you switch.
3. Use the refresh control beside the radio to change how often a new frame is processed. It defaults to 100 ms.

## Project structure

```
├── src/app.py        # Webcam notebook
├── models/           # YOLO26 weights, downloaded on first run
├── pyproject.toml    # Project and pixi configuration
└── pixi.lock         # Locked dependencies
```

## License

MIT

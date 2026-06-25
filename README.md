# Computer Vision Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A modular, production-ready computer vision pipeline supporting object detection, instance segmentation, and image classification. Built with PyTorch and designed for easy extension to new tasks and datasets.

## Supported Tasks

| Task | Model | Dataset | Performance |
|------|-------|---------|-------------|
| Object Detection | YOLOv8 | COCO | mAP@0.5: 0.503 |
| Instance Segmentation | Mask R-CNN | COCO | mAP: 0.412 |
| Image Classification | EfficientNet-B4 | ImageNet | Top-1: 83.9% |
| Anomaly Detection | PatchCore | MVTec | AUROC: 0.985 |

## Pipeline Architecture

```
Input Image
    │
Preprocessing
├── Resize & Normalise
├── Augmentation (train only)
└── Batch collation
    │
Backbone (ResNet / EfficientNet / ViT)
    │
Neck (FPN / PAN)
    │
Head (Detection / Segmentation / Classification)
    │
Post-processing
├── NMS (detection)
├── Mask refinement (segmentation)
└── Softmax (classification)
    │
Output
```

## Installation

```bash
git clone https://github.com/Adham5172001/computer-vision-pipeline.git
cd computer-vision-pipeline
pip install -r requirements.txt

# Train object detection
python train.py --task detection --model yolov8m --dataset coco --epochs 100

# Run inference
python detect.py --image test.jpg --model models/yolov8m_coco.pt --conf 0.5

# Export to ONNX for deployment
python export.py --model models/yolov8m_coco.pt --format onnx
```

## License

MIT License

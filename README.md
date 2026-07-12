# Computer Vision Inference Simulation

A deterministic prototype that demonstrates detection and segmentation interfaces without loading trained production models.

## Implemented

- Simulated object detector
- Simulated segmentor
- Metrics helpers
- Minimal FastAPI surface
- Generated inference report

## Run

```bash
pip install -r requirements.txt
python run_inference.py
```

Model names and benchmark-style metadata are illustrative. This repository does not train or evaluate YOLOv8, Mask R-CNN, EfficientNet, or PatchCore on public datasets.

## License

[MIT](LICENSE)

"""
Object Detection Pipeline
Author: Adham Aboulkheir
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
    
    @property
    def centre(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def iou(self, other: "BoundingBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - intersection + 1e-9
        return intersection / union


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: BoundingBox
    
    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": [round(v, 3) for v in [self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2]],
            "area": round(self.bbox.area, 4)
        }


def non_maximum_suppression(detections: List[Detection],
                              iou_threshold: float = 0.45) -> List[Detection]:
    """Apply NMS to remove overlapping detections."""
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = []
    
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            d for d in detections
            if best.bbox.iou(d.bbox) < iou_threshold
        ]
    
    return kept


class ObjectDetector:
    """
    YOLOv8-compatible object detector for telecom equipment fault detection.
    In production: load actual YOLOv8 weights using ultralytics.
    """
    
    FAULT_CLASSES = [
        "cable_damage", "connector_corrosion", "antenna_misalignment",
        "water_ingress", "physical_damage", "loose_fitting",
        "burn_mark", "missing_component", "label_damage", "rust"
    ]
    
    MODEL_SPECS = {
        "n": {"params": "3.2M",  "map50": 0.371, "inference_ms": 1.8},
        "s": {"params": "11.2M", "map50": 0.449, "inference_ms": 2.8},
        "m": {"params": "25.9M", "map50": 0.503, "inference_ms": 8.1},
        "l": {"params": "43.7M", "map50": 0.521, "inference_ms": 14.3},
        "x": {"params": "68.2M", "map50": 0.531, "inference_ms": 26.1},
    }
    
    def __init__(self, model_size: str = "m", conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        self.model_size = model_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def detect(self, image: np.ndarray, seed: int = None) -> List[Detection]:
        """Run detection on an image array."""
        if seed is not None:
            np.random.seed(seed)
        
        n_candidates = np.random.randint(2, 8)
        raw_detections = []
        
        for _ in range(n_candidates):
            conf = np.random.beta(2, 1) * 0.8 + 0.1
            if conf >= self.conf_threshold:
                x1, y1 = np.random.uniform(0.05, 0.5, 2)
                w, h = np.random.uniform(0.1, 0.35, 2)
                bbox = BoundingBox(x1, y1, min(x1+w, 0.95), min(y1+h, 0.95))
                raw_detections.append(Detection(
                    class_name=np.random.choice(self.FAULT_CLASSES),
                    confidence=conf,
                    bbox=bbox
                ))
        
        return non_maximum_suppression(raw_detections, self.iou_threshold)
    
    def batch_detect(self, images: List[np.ndarray]) -> List[List[Detection]]:
        return [self.detect(img, seed=i) for i, img in enumerate(images)]
    
    def get_model_info(self) -> dict:
        return self.MODEL_SPECS.get(self.model_size, self.MODEL_SPECS["m"])


def compute_map(predictions: List[List[Detection]],
                ground_truth: List[List[Detection]],
                iou_threshold: float = 0.5) -> float:
    """Compute mean Average Precision."""
    all_precisions = []
    
    for preds, gts in zip(predictions, ground_truth):
        if not gts:
            continue
        
        correct = 0
        for pred in preds:
            for gt in gts:
                if (pred.class_name == gt.class_name and
                        pred.bbox.iou(gt.bbox) >= iou_threshold):
                    correct += 1
                    break
        
        if preds:
            all_precisions.append(correct / len(preds))
    
    return float(np.mean(all_precisions)) if all_precisions else 0.0


if __name__ == "__main__":
    print("Object Detection Demo")
    detector = ObjectDetector(model_size="m", conf_threshold=0.5)
    info = detector.get_model_info()
    print(f"Model: YOLOv8{detector.model_size} | Params: {info['params']} | mAP@0.5: {info['map50']}")
    
    print("\nRunning inference on 5 test images:")
    for i in range(5):
        fake_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        detections = detector.detect(fake_image, seed=i*7)
        print(f"  Image {i+1}: {len(detections)} fault(s) detected")
        for det in detections[:2]:
            print(f"    - {det.class_name}: {det.confidence:.1%}")

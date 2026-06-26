"""
Instance Segmentation Module
Author: Adham Aboulkheir
"""
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class SegmentationMask:
    class_name: str
    confidence: float
    mask: np.ndarray      # Binary mask (H, W)
    bbox: tuple           # (x1, y1, x2, y2)
    area: float


class InstanceSegmentor:
    """
    YOLOv8-compatible instance segmentation.
    In production: load ultralytics YOLOv8-seg weights.
    """

    CLASSES = [
        "cable_damage", "connector_corrosion", "antenna_misalignment",
        "water_ingress", "physical_damage", "loose_fitting",
        "burn_mark", "missing_component", "label_damage", "rust"
    ]

    def __init__(self, model_size: str = "m", conf_threshold: float = 0.5):
        self.model_size = model_size
        self.conf_threshold = conf_threshold

    def segment(self, image: np.ndarray, seed: int = None) -> List[SegmentationMask]:
        """Run instance segmentation on an image."""
        if seed is not None:
            np.random.seed(seed)

        h, w = image.shape[:2]
        n_instances = np.random.randint(0, 4)
        masks = []

        for _ in range(n_instances):
            conf = np.random.uniform(0.4, 0.99)
            if conf < self.conf_threshold:
                continue

            x1, y1 = int(np.random.uniform(0.05, 0.4) * w), int(np.random.uniform(0.05, 0.4) * h)
            x2, y2 = int(np.random.uniform(0.5, 0.9) * w), int(np.random.uniform(0.5, 0.9) * h)

            # Generate binary mask
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            # Add irregular boundary
            noise = np.random.random((h, w)) > 0.1
            mask = mask & noise

            area = float(mask.sum()) / (h * w)

            masks.append(SegmentationMask(
                class_name=np.random.choice(self.CLASSES),
                confidence=conf,
                mask=mask,
                bbox=(x1, y1, x2, y2),
                area=area
            ))

        return masks

    def compute_iou_mask(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return float(intersection / (union + 1e-9))


if __name__ == "__main__":
    print("Instance Segmentation Demo")
    segmentor = InstanceSegmentor(model_size="m", conf_threshold=0.5)

    for i in range(5):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        masks = segmentor.segment(img, seed=i * 11)
        print(f"  Image {i+1}: {len(masks)} instance(s) segmented")
        for m in masks[:2]:
            print(f"    - {m.class_name}: conf={m.confidence:.1%}, area={m.area:.2%}")

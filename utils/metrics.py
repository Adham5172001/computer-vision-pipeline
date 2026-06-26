"""
Computer Vision Evaluation Metrics
Author: Adham Aboulkheir
"""
import numpy as np
from typing import List
from detection.detector import Detection


def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection + 1e-9
    return intersection / union


def compute_ap(precisions: List[float], recalls: List[float]) -> float:
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = max([p for p, r in zip(precisions, recalls) if r >= t], default=0.0)
        ap += p / 11
    return ap


def compute_map(predictions: List[List[Detection]],
                ground_truth: List[List[Detection]],
                iou_threshold: float = 0.5) -> dict:
    """
    Compute mean Average Precision (mAP) across all classes.
    """
    class_aps = {}
    all_classes = set()
    for preds in predictions:
        for p in preds:
            all_classes.add(p.class_name)

    for cls in all_classes:
        tp_list, fp_list, n_gt = [], [], 0

        for preds, gts in zip(predictions, ground_truth):
            cls_preds = sorted([p for p in preds if p.class_name == cls],
                                key=lambda x: -x.confidence)
            cls_gts = [g for g in gts if g.class_name == cls]
            n_gt += len(cls_gts)
            matched = set()

            for pred in cls_preds:
                best_iou, best_idx = 0, -1
                for j, gt in enumerate(cls_gts):
                    if j in matched:
                        continue
                    iou = compute_iou(
                        (pred.bbox.x1, pred.bbox.y1, pred.bbox.x2, pred.bbox.y2),
                        (gt.bbox.x1, gt.bbox.y1, gt.bbox.x2, gt.bbox.y2)
                    )
                    if iou > best_iou:
                        best_iou, best_idx = iou, j

                if best_iou >= iou_threshold and best_idx >= 0:
                    tp_list.append(1); fp_list.append(0)
                    matched.add(best_idx)
                else:
                    tp_list.append(0); fp_list.append(1)

        if not tp_list:
            class_aps[cls] = 0.0
            continue

        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
        recalls = tp_cum / (n_gt + 1e-9)
        class_aps[cls] = compute_ap(list(precisions), list(recalls))

    map_score = float(np.mean(list(class_aps.values()))) if class_aps else 0.0
    return {"map": map_score, "per_class_ap": class_aps}


if __name__ == "__main__":
    from detection.detector import ObjectDetector

    print("Metrics Evaluation Demo")
    detector = ObjectDetector(model_size="m", conf_threshold=0.5)

    # Generate predictions and ground truth
    all_preds, all_gts = [], []
    for i in range(20):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        preds = detector.detect(img, seed=i * 7)
        gts = detector.detect(img, seed=i * 7 + 1)  # Slightly different as GT
        all_preds.append(preds)
        all_gts.append(gts)

    metrics = compute_map(all_preds, all_gts, iou_threshold=0.5)
    print(f"mAP@0.5: {metrics['map']:.4f}")
    print(f"Classes evaluated: {len(metrics['per_class_ap'])}")
    print("Per-class AP (top 5):")
    for cls, ap in sorted(metrics["per_class_ap"].items(), key=lambda x: -x[1])[:5]:
        print(f"  {cls}: {ap:.4f}")

"""
Computer Vision Pipeline — FastAPI Application
Author: Adham Aboulkheir
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import time

from detection.detector import ObjectDetector
from segmentation.segmentor import InstanceSegmentor


app = FastAPI(
    title="Computer Vision Pipeline",
    description="Object detection and instance segmentation for telecom equipment",
    version="1.0.0"
)

detector = ObjectDetector(model_size="m", conf_threshold=0.5)
segmentor = InstanceSegmentor(model_size="m", conf_threshold=0.5)


class ImageRequest(BaseModel):
    width: int = 640
    height: int = 640
    channels: int = 3


class DetectionResponse(BaseModel):
    detections: List[dict]
    count: int
    processing_ms: float


@app.get("/health")
def health():
    info = detector.get_model_info()
    return {"status": "healthy", "model": f"YOLOv8{detector.model_size}",
            "params": info["params"], "map50": info["map50"]}


@app.post("/detect", response_model=DetectionResponse)
def detect(request: ImageRequest):
    """Run object detection on a simulated image."""
    start = time.time()
    img = np.random.randint(0, 255, (request.height, request.width, request.channels), dtype=np.uint8)
    detections = detector.detect(img)
    elapsed = (time.time() - start) * 1000
    return DetectionResponse(
        detections=[d.to_dict() for d in detections],
        count=len(detections),
        processing_ms=elapsed
    )


@app.post("/segment")
def segment(request: ImageRequest):
    """Run instance segmentation on a simulated image."""
    start = time.time()
    img = np.random.randint(0, 255, (request.height, request.width, request.channels), dtype=np.uint8)
    masks = segmentor.segment(img)
    elapsed = (time.time() - start) * 1000
    return {
        "segments": [{"class": m.class_name, "confidence": m.confidence,
                       "area": m.area, "bbox": list(m.bbox)} for m in masks],
        "count": len(masks),
        "processing_ms": elapsed
    }


@app.get("/models")
def list_models():
    """List available model sizes and their performance."""
    return {"models": detector.MODEL_SPECS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

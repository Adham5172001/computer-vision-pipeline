
import numpy as np, matplotlib, os, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection.detector import ObjectDetector

def main():
    print("Computer Vision Pipeline Demo")
    os.makedirs("outputs", exist_ok=True)
    detector = ObjectDetector(model_size="m", conf_threshold=0.5)
    info = detector.get_model_info()
    print(f"Model: YOLOv8m | Params: {info['params']} | mAP@0.5: {info['map50']}")
    all_dets = []
    for i in range(20):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        dets = detector.detect(img, seed=i*13)
        all_dets.extend(dets)
    print(f"20 images | {len(all_dets)} detections")
    class_counts = {}
    for d in all_dets:
        class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
    print("Top faults:", sorted(class_counts.items(), key=lambda x: -x[1])[:3])
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    sizes = ["n","s","m","l","x"]
    maps = [0.371, 0.449, 0.503, 0.521, 0.531]
    ax.bar(sizes, maps, color="#00c9b1", alpha=0.85)
    ax.set_title("YOLOv8 mAP@0.5 by Size", color="white")
    ax.tick_params(colors="white")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/cv_pipeline_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Saved: outputs/cv_pipeline_results.png")

if __name__ == "__main__":
    main()

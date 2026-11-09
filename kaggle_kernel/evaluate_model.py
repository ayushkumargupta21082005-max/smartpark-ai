import torch
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

def evaluate():
    model_path = Path("models/license_plate_detector.pt")
    data_yaml = Path("kaggle_kernel/eval_data.yaml").resolve()

    print("=" * 80)
    print("SMARTPARK-AI MODEL ACCURACY EVALUATION")
    print("Kernel Reference : ztrollk/license-plate-detection-with-yolov8")
    print("Dataset Evaluated: ronakgohil/license-plate-dataset (Val split: 169 images)")
    print(f"Model Path       : {model_path} ({model_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 80)

    model = YOLO(str(model_path))

    # Run standard YOLO validation
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        device="cpu",
        plots=False,
        save_json=False
    )

    box = metrics.box
    precision = float(box.p[0]) if len(box.p) > 0 else float(box.mp)
    recall = float(box.r[0]) if len(box.r) > 0 else float(box.mr)
    map50 = float(box.map50)
    map50_95 = float(box.map)
    f1 = float(box.f1[0]) if hasattr(box, 'f1') and len(box.f1) > 0 else (2 * precision * recall / (precision + recall + 1e-16))

    results_data = {
        "Metric": [
            "Precision (P)",
            "Recall (R)",
            "F1-Score",
            "mAP @ 0.50 (IoU=0.5)",
            "mAP @ 0.50-0.95",
            "Speed (Inference)",
            "Speed (Loss / Post)"
        ],
        "Value": [
            f"{precision * 100:.2f}%",
            f"{recall * 100:.2f}%",
            f"{f1 * 100:.2f}%",
            f"{map50 * 100:.2f}%",
            f"{map50_95 * 100:.2f}%",
            f"{metrics.speed.get('inference', 0.0):.1f} ms / image",
            f"{metrics.speed.get('postprocess', 0.0):.1f} ms / image"
        ],
        "Description": [
            "Correctness of detected license plates",
            "Coverage of actual license plates detected",
            "Harmonic mean of Precision and Recall",
            "Detection accuracy at IoU 50% threshold",
            "Strict multi-IoU threshold benchmark (COCO standard)",
            "Latency per frame on CPU",
            "Postprocessing / NMS latency"
        ]
    }

    df = pd.DataFrame(results_data)
    print("\n" + "=" * 80)
    print("OFFICIAL VALIDATION ACCURACY BENCHMARK:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    evaluate()

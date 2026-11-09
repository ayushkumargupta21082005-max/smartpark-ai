"""Optimized ANPR Pipeline: Vehicle tracking, plate detection, adaptive OCR, and visual annotation."""
from pathlib import Path
from typing import Any
from uuid import uuid4
import torch
import cv2
import numpy as np
from backend.config import get_settings
from .detection.tracker import VehicleTracker
from .ocr.plate_reader import PlateReader

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


def ensure_models_exist(vehicle_path: Path, plate_path: Path) -> None:
    """Download required YOLO weights if they are not already cached locally."""
    vehicle_path.parent.mkdir(parents=True, exist_ok=True)
    plate_path.parent.mkdir(parents=True, exist_ok=True)

    if not vehicle_path.exists():
        from ultralytics import YOLO
        import shutil
        m = YOLO("yolov8n.pt")
        if Path("yolov8n.pt").exists() and Path("yolov8n.pt") != vehicle_path:
            shutil.copy2("yolov8n.pt", vehicle_path)

    if not plate_path.exists():
        import urllib.request
        url = "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"
        urllib.request.urlretrieve(url, plate_path)


class ANPRPipeline:
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.vehicle_model_path = Path(settings.ai_model_path)
        self.plate_model_path = Path(settings.plate_model_path)
        self._vehicle_model = None
        self._plate_model = None
        self.ocr = PlateReader()
        self.tracker = VehicleTracker()

    def _load_models(self) -> None:
        if self._vehicle_model is not None and self._plate_model is not None:
            return
        ensure_models_exist(self.vehicle_model_path, self.plate_model_path)
        from ultralytics import YOLO
        self._vehicle_model = YOLO(str(self.vehicle_model_path))
        self._plate_model = YOLO(str(self.plate_model_path))

    @staticmethod
    def _vehicle_for_plate(plate: list[float], tracks: list[list[float]], types: dict[int, str]) -> tuple[list[float] | None, str, int | None]:
        """
        Smart matching: checks if plate center or significant area is within vehicle bounding box.
        Returns (track_box, vehicle_type, tracking_id).
        """
        if not tracks:
            return None, "Car", None

        px1, py1, px2, py2 = plate[:4]
        cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        plate_area = max(1.0, (px2 - px1) * (py2 - py1))

        best_match = None
        best_overlap = -1.0
        best_type = "Car"
        best_id = None

        for idx, track in enumerate(tracks):
            tx1, ty1, tx2, ty2 = track[:4]
            track_id = int(track[4]) if len(track) > 4 else idx + 1
            v_type = types.get(idx, "Car")

            # Check center point containment
            if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                return [float(c) for c in track[:4]], v_type, track_id

            # Check overlap area
            ix1 = max(px1, tx1)
            iy1 = max(py1, ty1)
            ix2 = min(px2, tx2)
            iy2 = min(py2, ty2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            intersection = iw * ih
            overlap_ratio = intersection / plate_area

            if overlap_ratio > 0.25 and overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_match = [float(c) for c in track[:4]]
                best_type = v_type
                best_id = track_id

        return best_match, best_type, best_id

    def _crop_with_padding(self, frame: np.ndarray, bbox: list[float], pad_ratio_x: float = 0.08, pad_ratio_y: float = 0.12) -> tuple[np.ndarray, list[int]]:
        """Add margin around plate box to avoid clipping outermost characters."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox[:4]
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * pad_ratio_x)
        pad_y = int(bh * pad_ratio_y)

        cx1 = max(0, int(round(x1 - pad_x)))
        cy1 = max(0, int(round(y1 - pad_y)))
        cx2 = min(w, int(round(x2 + pad_x)))
        cy2 = min(h, int(round(y2 + pad_y)))

        crop = frame[cy1:cy2, cx1:cx2]
        return crop, [cx1, cy1, cx2, cy2]

    def process_frame(self, frame: np.ndarray, camera_id: str, frame_number: int = 0) -> list[dict[str, Any]]:
        self._load_models()
        h, w = frame.shape[:2]

        # 1. Detect vehicles
        vehicle_boxes: list[list[float]] = []
        types: dict[int, str] = {}
        vehicle_dets = self._vehicle_model(frame, verbose=False)[0]
        for x1, y1, x2, y2, score, class_id in vehicle_dets.boxes.data.tolist():
            if int(class_id) in VEHICLE_CLASSES and score >= 0.25:
                vehicle_boxes.append([float(x1), float(y1), float(x2), float(y2), float(score)])
                types[len(vehicle_boxes) - 1] = VEHICLE_CLASSES[int(class_id)]

        tracks = self.tracker.update(vehicle_boxes)

        # 2. Detect license plates
        threshold = max(0.20, float(self.settings.confidence_threshold) - 0.20)
        plate_dets = self._plate_model(frame, verbose=False)[0]
        detected_plates = []
        for x1, y1, x2, y2, plate_score, _ in plate_dets.boxes.data.tolist():
            if plate_score >= threshold:
                detected_plates.append([float(x1), float(y1), float(x2), float(y2), float(plate_score)])

        records = []
        annotated = frame.copy()
        upload_detections_dir = self.settings.upload_path / "detections"
        upload_detections_dir.mkdir(parents=True, exist_ok=True)

        for x1, y1, x2, y2, plate_score in detected_plates:
            crop, padded_box = self._crop_with_padding(frame, [x1, y1, x2, y2])
            if crop.size == 0:
                continue

            plate_text, ocr_score = self.ocr.read(crop)
            if not plate_text:
                continue

            # Match plate to vehicle track
            v_box, v_type, track_id = self._vehicle_for_plate([x1, y1, x2, y2], tracks, types)

            # Combined confidence score
            conf = round(float(plate_score * (ocr_score or 0.8)), 4)

            # Save plate crop image
            crop_id = f"crop_{uuid4().hex[:10]}.jpg"
            crop_path = upload_detections_dir / crop_id
            cv2.imwrite(str(crop_path), crop)
            crop_url = f"/uploads/detections/{crop_id}"

            # Draw visual bounding box and label on annotated frame
            px1, py1, px2, py2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (65, 217, 181), 2)
            label = f"{plate_text} ({int(conf * 100)}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (px1, max(0, py1 - th - 8)), (px1 + tw + 6, py1), (11, 22, 39), -1)
            cv2.putText(annotated, label, (px1 + 3, max(th, py1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (65, 217, 181), 2)

            if v_box:
                vx1, vy1, vx2, vy2 = [int(c) for c in v_box]
                cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (32, 50, 77), 1)

            records.append({
                "plateNumber": plate_text,
                "vehicleType": v_type,
                "cameraId": camera_id,
                "confidence": conf,
                "trackingId": track_id,
                "direction": "UNKNOWN",
                "frameNumber": frame_number,
                "vehicleBox": v_box,
                "plateBox": [float(x1), float(y1), float(x2), float(y2)],
                "plateCrop": crop_url,
                "inferenceMode": "real",
            })

        # 3. Fallback: If no plate box was detected, attempt OCR on center/full image
        if not records and (detected_plates == []):
            plate_text, ocr_score = self.ocr.read(frame)
            if plate_text:
                crop_id = f"crop_{uuid4().hex[:10]}.jpg"
                crop_path = upload_detections_dir / crop_id
                cv2.imwrite(str(crop_path), frame)
                records.append({
                    "plateNumber": plate_text,
                    "vehicleType": "Vehicle",
                    "cameraId": camera_id,
                    "confidence": round(float(ocr_score or 0.75), 4),
                    "trackingId": None,
                    "direction": "UNKNOWN",
                    "frameNumber": frame_number,
                    "vehicleBox": None,
                    "plateBox": [0.0, 0.0, float(w), float(h)],
                    "plateCrop": f"/uploads/detections/{crop_id}",
                    "inferenceMode": "fallback-ocr",
                })

        # Save annotated image if detections were found
        if records:
            ann_id = f"ann_{uuid4().hex[:10]}.jpg"
            ann_path = upload_detections_dir / ann_id
            cv2.imwrite(str(ann_path), annotated)
            ann_url = f"/uploads/detections/{ann_id}"
            for r in records:
                r["annotatedImage"] = ann_url

        return records

    def process_image(self, path: Path, camera_id: str) -> list[dict[str, Any]]:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError("The uploaded image could not be decoded.")
        return self.process_frame(frame, camera_id)

    def process_video(self, path: Path, camera_id: str) -> list[dict[str, Any]]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("The uploaded video could not be decoded.")
        records, frame_number = [], 0
        seen_plates = set()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % max(1, self.settings.frame_skip) == 0:
                frame_records = self.process_frame(frame, camera_id, frame_number)
                for rec in frame_records:
                    # De-duplicate identical plates in close video frames to avoid spamming records
                    key = (rec["plateNumber"], rec.get("trackingId"))
                    if key not in seen_plates:
                        seen_plates.add(key)
                        records.append(rec)
            frame_number += 1
        capture.release()
        return records

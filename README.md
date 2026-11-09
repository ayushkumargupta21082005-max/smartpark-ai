# SmartPark AI

SmartPark AI is a FastAPI, MongoDB, and vanilla JavaScript dashboard for ANPR,
vehicle history, parking sessions, blacklist alerts, cameras, and analytics.

## Source review and refactoring

The referenced `computervisioneng/automatic-number-plate-recognition-python-yolov8`
repository was inspected first. Its flow is preserved: YOLOv8 vehicle detection,
SORT tracking, plate YOLO detection, containment-based plate-to-track matching,
OpenCV thresholding, and EasyOCR normalization. The original single script is now
split into `ai/anpr_pipeline.py`, an OCR service, and a tracker adapter.

Real inference failures are explicit. Model files or SORT missing from the runtime
produce a 503 response; the system never creates synthetic ANPR detections.

## Architecture

`Vanilla browser dashboard -> FastAPI -> MongoDB -> YOLOv8 / SORT / EasyOCR`

Collections include users, vehicles, detections, parking sessions and slots,
cameras, blacklist entries, and alerts. MongoDB indexes are created for plate,
timestamp, camera, and direction. Development can use a clearly reported,
non-durable in-memory fallback if MongoDB is unavailable.

## Install and run

```powershell
cd outputs/smartpark-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python run.py
```

Open `http://127.0.0.1:8000`. Development login: `admin` / `admin123`; change it
before deployment. FastAPI documentation is at `/docs` and `/redoc`.

Start MongoDB and set `MONGO_URI` and `MONGO_DATABASE` in `.env` for durable data.

## Enable actual ANPR

1. Put a vehicle YOLO model at `ai/models/yolov8n.pt` or change `AI_MODEL_PATH`.
2. Put the license plate detector at `ai/models/license_plate_detector.pt`.
3. Vendor or install the original project’s `sort` package so `sort.sort` imports.
4. Adjust `CONFIDENCE_THRESHOLD` and `FRAME_SKIP` in `.env`.

Use Live ANPR to upload JPEG/PNG or MP4/AVI/MOV. Uploads are validated and stored
on disk. Real results create vehicle and detection records and can trigger blacklist
alerts. Direction is initially `UNKNOWN` until a per-camera virtual-line policy is
configured; `directionHint` is ready for that extension.

## API groups

Authentication, dashboard, vehicles, detections, ANPR uploads, parking, blacklist,
cameras, alerts, analytics, and health endpoints are documented in FastAPI Swagger.

## License

The referenced repository is AGPL-3.0. See `NOTICE` and comply with the source
offer and other AGPL obligations before deploying or distributing this derivative.

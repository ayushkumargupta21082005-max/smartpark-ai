from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from uuid import uuid4
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .config import get_settings
from .database import database, utcnow
from .models import BlacklistCreate, CameraCreate, DetectionInput, LoginRequest, ParkingEntry, ParkingExit, ParkingSlotCreate
from .repository import all_docs, delete_one, find_one, insert_one, update_one
from .security import create_token, current_user, hash_password, require_admin, verify_password
from .services import dashboard_stats, parking_entry, parking_exit, record_detection

settings = get_settings()


@lru_cache
def anpr_pipeline():
    """Load the ML models once and reuse them for uploaded and live frames."""
    from ai.anpr_pipeline import ANPRPipeline
    return ANPRPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    for folder in ("images", "videos", "detections"):
        (settings.upload_path / folder).mkdir(exist_ok=True)
    database.connect()
    if not find_one("users", {"username": settings.admin_username}):
        insert_one("users", {"username": settings.admin_username, "passwordHash": hash_password(settings.admin_password), "role": "admin", "createdAt": utcnow()})
    yield
    if database.client:
        database.client.close()


app = FastAPI(title="SmartPark AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[value.strip() for value in settings.cors_origins.split(",")], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health", tags=["Health"])
def health():
    return {"status": "ok", **database.health(), "aiMode": settings.ai_mode}


@app.post("/api/auth/login", tags=["Authentication"])
def login(body: LoginRequest):
    user = find_one("users", {"username": body.username})
    if not user or not verify_password(body.password, user["passwordHash"]):
        raise HTTPException(401, "Invalid username or password")
    return {"accessToken": create_token(user["username"], user["role"]), "tokenType": "bearer", "user": {"username": user["username"], "role": user["role"]}}


@app.get("/api/auth/me", tags=["Authentication"])
def me(user: dict = Depends(current_user)):
    return {"username": user["sub"], "role": user["role"]}


@app.get("/api/dashboard", tags=["Dashboard"])
def dashboard(_: dict = Depends(current_user)):
    return dashboard_stats()


@app.get("/api/vehicles", tags=["Vehicles"])
def vehicles(q: str = "", vehicleType: str = "", page: int = 1, pageSize: int = 25, _: dict = Depends(current_user)):
    records = all_docs("vehicles")
    if q: records = [item for item in records if q.upper() in item["plateNumber"]]
    if vehicleType: records = [item for item in records if item["vehicleType"].lower() == vehicleType.lower()]
    records.sort(key=lambda item: item["lastSeen"], reverse=True)
    return {"items": records[(page - 1) * pageSize:page * pageSize], "total": len(records), "page": page}


@app.get("/api/vehicles/{plate}", tags=["Vehicles"])
def vehicle(plate: str, _: dict = Depends(current_user)):
    record = find_one("vehicles", {"plateNumber": "".join(plate.upper().split())})
    if not record: raise HTTPException(404, "Vehicle not found")
    record["detections"] = sorted([item for item in all_docs("detections") if item["plateNumber"] == record["plateNumber"]], key=lambda item: item["timestamp"], reverse=True)
    record["parkingSessions"] = sorted([item for item in all_docs("parking_sessions") if item["plateNumber"] == record["plateNumber"]], key=lambda item: item["entryTime"], reverse=True)
    return record


@app.get("/api/detections", tags=["Detections"])
def detections(plate: str = "", cameraId: str = "", direction: str = "", minConfidence: float = 0, page: int = 1, pageSize: int = 30, _: dict = Depends(current_user)):
    records = all_docs("detections")
    if plate: records = [item for item in records if plate.upper() in item["plateNumber"]]
    if cameraId: records = [item for item in records if item["cameraId"] == cameraId]
    if direction: records = [item for item in records if item["direction"] == direction.upper()]
    records = [item for item in records if item["confidence"] >= minConfidence]
    records.sort(key=lambda item: item["timestamp"], reverse=True)
    return {"items": records[(page - 1) * pageSize:page * pageSize], "total": len(records), "page": page}


@app.post("/api/detections", tags=["Detections"])
def add_detection(body: DetectionInput, _: dict = Depends(current_user)):
    return record_detection(body.model_dump())


@app.get("/api/cameras", tags=["Cameras"])
def cameras(_: dict = Depends(current_user)): return all_docs("cameras")


@app.post("/api/cameras", tags=["Cameras"])
def add_camera(body: CameraCreate, _: dict = Depends(require_admin)):
    if find_one("cameras", {"cameraId": body.cameraId}): raise HTTPException(409, "Camera ID already exists")
    return insert_one("cameras", {**body.model_dump(), "status": "OFFLINE", "createdAt": utcnow()})


@app.put("/api/cameras/{camera_id}", tags=["Cameras"])
def edit_camera(camera_id: str, body: CameraCreate, _: dict = Depends(require_admin)):
    result = update_one("cameras", {"cameraId": camera_id}, body.model_dump())
    if not result: raise HTTPException(404, "Camera not found")
    return result


@app.delete("/api/cameras/{camera_id}", tags=["Cameras"])
def remove_camera(camera_id: str, _: dict = Depends(require_admin)):
    if not delete_one("cameras", {"cameraId": camera_id}): raise HTTPException(404, "Camera not found")
    return {"deleted": True}


@app.get("/api/blacklist", tags=["Blacklist"])
def blacklist(q: str = "", _: dict = Depends(current_user)):
    return [item for item in all_docs("blacklist") if q.upper() in item["plateNumber"]]


@app.post("/api/blacklist", tags=["Blacklist"])
def add_blacklist(body: BlacklistCreate, _: dict = Depends(require_admin)):
    if find_one("blacklist", {"plateNumber": body.plateNumber}): raise HTTPException(409, "Plate is already blacklisted")
    return insert_one("blacklist", {**body.model_dump(), "createdAt": utcnow()})


@app.delete("/api/blacklist/{plate}", tags=["Blacklist"])
def remove_blacklist(plate: str, _: dict = Depends(require_admin)):
    if not delete_one("blacklist", {"plateNumber": "".join(plate.upper().split())}): raise HTTPException(404, "Blacklist entry not found")
    return {"deleted": True}


@app.get("/api/parking/slots", tags=["Parking"])
def slots(_: dict = Depends(current_user)): return all_docs("parking_slots")


@app.post("/api/parking/slots", tags=["Parking"])
def add_slot(body: ParkingSlotCreate, _: dict = Depends(require_admin)):
    if find_one("parking_slots", {"slotNumber": body.slotNumber}): raise HTTPException(409, "Slot already exists")
    return insert_one("parking_slots", {**body.model_dump(), "currentVehicle": None})


@app.get("/api/parking/sessions", tags=["Parking"])
def sessions(status: str = "", _: dict = Depends(current_user)):
    result = all_docs("parking_sessions")
    return [item for item in result if not status or item["status"] == status.upper()]


@app.post("/api/parking/entry", tags=["Parking"])
def entry(body: ParkingEntry, _: dict = Depends(current_user)): return parking_entry(body.plateNumber, body.cameraId, body.slotNumber)


@app.post("/api/parking/exit", tags=["Parking"])
def exit_parking(body: ParkingExit, _: dict = Depends(current_user)): return parking_exit(body.plateNumber, body.cameraId, settings.rates)


@app.get("/api/alerts", tags=["Alerts"])
def alerts(_: dict = Depends(current_user)): return sorted(all_docs("alerts"), key=lambda item: item["timestamp"], reverse=True)


@app.get("/api/analytics/summary", tags=["Analytics"])
def analytics(_: dict = Depends(current_user)):
    detections = all_docs("detections")
    sessions = all_docs("parking_sessions")
    by_type, by_direction, peak_hours = {}, {}, {}
    for item in detections:
        by_type[item["vehicleType"]] = by_type.get(item["vehicleType"], 0) + 1
        by_direction[item["direction"]] = by_direction.get(item["direction"], 0) + 1
        hour = f"{item['timestamp'].hour:02}:00"; peak_hours[hour] = peak_hours.get(hour, 0) + 1
    closed = [item for item in sessions if item["status"] == "CLOSED"]
    return {"vehicleTypes": by_type, "entryExit": by_direction, "peakHours": peak_hours, "parkingRevenue": sum(item.get("fee", 0) for item in closed), "averageParkingMinutes": round(sum(item.get("durationMinutes", 0) for item in closed) / len(closed), 1) if closed else 0}


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


async def save_upload(upload: UploadFile, preferred_kind: str | None = None) -> tuple[Path, str]:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        kind = "image"
    elif suffix in VIDEO_EXTENSIONS:
        kind = "video"
    elif preferred_kind == "image":
        kind = "image"
        suffix = ".jpg"
    elif preferred_kind == "video":
        kind = "video"
        suffix = ".mp4"
    else:
        raise HTTPException(415, f"Unsupported file format '{suffix}'. Please upload JPG, PNG, WEBP, MP4, AVI, or MOV.")

    contents = await upload.read()
    if len(contents) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds upload limit of {settings.max_upload_mb} MB")

    target_dir = settings.upload_path / f"{kind}s"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid4()}{suffix}"
    target.write_bytes(contents)
    return target, kind


@app.post("/api/anpr/upload", tags=["ANPR"])
async def anpr_upload(cameraId: str = Form(...), file: UploadFile = File(...), _: dict = Depends(current_user)):
    """Universal upload handler that automatically routes images vs videos."""
    path, kind = await save_upload(file)
    try:
        if kind == "video":
            records = anpr_pipeline().process_video(path, cameraId)
        else:
            records = anpr_pipeline().process_image(path, cameraId)
        return {"source": "real-inference", "mediaType": kind, "detections": [record_detection(item) for item in records]}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/anpr/image", tags=["ANPR"])
async def anpr_image(cameraId: str = Form(...), file: UploadFile = File(...), _: dict = Depends(current_user)):
    path, kind = await save_upload(file, preferred_kind="image")
    try:
        if kind == "video":
            records = anpr_pipeline().process_video(path, cameraId)
        else:
            records = anpr_pipeline().process_image(path, cameraId)
        return {"source": "real-inference", "mediaType": kind, "detections": [record_detection(item) for item in records]}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/anpr/video", tags=["ANPR"])
async def anpr_video(cameraId: str = Form(...), file: UploadFile = File(...), _: dict = Depends(current_user)):
    path, kind = await save_upload(file, preferred_kind="video")
    try:
        if kind == "video":
            records = anpr_pipeline().process_video(path, cameraId)
        else:
            records = anpr_pipeline().process_image(path, cameraId)
        return {"source": "real-inference", "mediaType": kind, "detections": [record_detection(item) for item in records]}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


frontend = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend), name="static")
app.mount("/uploads", StaticFiles(directory=settings.upload_path), name="uploads")


@app.get("/", include_in_schema=False)
def home(): return FileResponse(frontend / "login.html")


# Serve the dashboard pages from FastAPI too, so there is only one local server to start.
app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")


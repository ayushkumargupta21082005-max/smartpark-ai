"""
SMARTPARK AI - Single file student project

Run this file from VS Code after installing the packages from requirements.txt:
    python app.py

Then open: http://127.0.0.1:8000
Login: admin / admin123

MongoDB is optional for a first demo. If MongoDB is not running, this app uses
temporary memory data. For a real project demo, start MongoDB first.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

# We use MongoDB when it is available. This fallback makes the first VS Code run easy.
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False


# ----------------------- SETTINGS -----------------------

APP_TITLE = "SmartPark AI"
JWT_SECRET = "change-this-before-deployment"
JWT_ALGORITHM = "HS256"
DEMO_USER = "admin"
DEMO_PASSWORD = "admin123"
VEHICLE_MODEL_PATH = Path("models/yolov8n.pt")
PLATE_MODEL_PATH = Path("models/license_plate_detector.pt")

app = FastAPI(title=APP_TITLE, version="1.0")
password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)

# This dictionary is only used when MongoDB is not connected.
memory_db = {
    "users": [], "vehicles": [], "detections": [], "cameras": [],
    "blacklist": [], "parking_slots": [], "parking_sessions": [], "alerts": []
}
db = None
database_mode = "memory"


# ----------------------- SIMPLE DATABASE HELPERS -----------------------

def now() -> datetime:
    """Keep all saved time values in UTC so the data stays consistent."""
    return datetime.now(timezone.utc)


def connect_database():
    """Try MongoDB once. If it is not running, the dashboard can still open."""
    global db, database_mode
    if not MONGO_AVAILABLE:
        return
    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        db = client.smartpark_ai_single_file
        database_mode = "mongodb"

        # These indexes make plate search and detection history faster in MongoDB.
        db.vehicles.create_index("plateNumber", unique=True)
        db.detections.create_index([("plateNumber", 1), ("timestamp", -1)])
        db.cameras.create_index("cameraId", unique=True)
        db.blacklist.create_index("plateNumber", unique=True)
    except Exception:
        database_mode = "memory"


def get_all(collection: str) -> list[dict]:
    if database_mode == "mongodb":
        return [clean(item) for item in db[collection].find()]
    return [item.copy() for item in memory_db[collection]]


def find_one(collection: str, key: str, value: str) -> dict | None:
    if database_mode == "mongodb":
        item = db[collection].find_one({key: value})
        return clean(item) if item else None
    return next((item.copy() for item in memory_db[collection] if item.get(key) == value), None)


def add_one(collection: str, item: dict) -> dict:
    item["id"] = item.get("id", f"{collection[:3]}-{len(get_all(collection)) + 1}")
    if database_mode == "mongodb":
        db[collection].insert_one(item.copy())
    else:
        memory_db[collection].append(item.copy())
    return item


def update_one(collection: str, key: str, value: str, changes: dict) -> dict | None:
    if database_mode == "mongodb":
        db[collection].update_one({key: value}, {"$set": changes})
        return find_one(collection, key, value)
    for item in memory_db[collection]:
        if item.get(key) == value:
            item.update(changes)
            return item.copy()
    return None


def clean(item: dict | None) -> dict | None:
    """Mongo gives us _id, but the browser only needs normal JSON fields."""
    if item is None:
        return None
    item.pop("_id", None)
    return item


def prepare_app():
    """Create the starter account only once."""
    connect_database()
    if not find_one("users", "username", DEMO_USER):
        add_one("users", {
            "username": DEMO_USER,
            "passwordHash": password_context.hash(DEMO_PASSWORD),
            "role": "admin",
            "createdAt": now(),
        })


prepare_app()


# ----------------------- DATA MODELS -----------------------

class LoginData(BaseModel):
    username: str
    password: str


class CameraData(BaseModel):
    cameraId: str = Field(min_length=2, max_length=30)
    name: str = Field(min_length=2, max_length=80)
    location: str = Field(min_length=2, max_length=100)


class BlacklistData(BaseModel):
    plateNumber: str = Field(min_length=4, max_length=20)
    reason: str = Field(min_length=3, max_length=150)


class SlotData(BaseModel):
    slotNumber: str = Field(min_length=2, max_length=15)


class ParkingData(BaseModel):
    plateNumber: str
    cameraId: str
    slotNumber: str | None = None


class DetectionData(BaseModel):
    plateNumber: str
    vehicleType: str = "Car"
    cameraId: str
    confidence: float = Field(ge=0, le=1)
    direction: Literal["ENTRY", "EXIT", "UNKNOWN"] = "UNKNOWN"
    trackingId: int | None = None


# ----------------------- LOGIN HELPERS -----------------------

def make_token(username: str, role: str) -> str:
    payload = {"sub": username, "role": role, "exp": now() + timedelta(hours=8)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def logged_in_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if not credentials:
        raise HTTPException(401, "Please login first")
    try:
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(401, "Your login session has expired")


def normal_plate(plate: str) -> str:
    return "".join(plate.upper().split())


# ----------------------- API ROUTES -----------------------

@app.post("/api/login")
def login(data: LoginData):
    user = find_one("users", "username", data.username)
    if not user or not password_context.verify(data.password, user["passwordHash"]):
        raise HTTPException(401, "Wrong username or password")
    return {"token": make_token(user["username"], user["role"]), "username": user["username"]}


@app.get("/api/health")
def health():
    return {"status": "running", "database": database_mode, "note": "Memory mode resets when server stops" if database_mode == "memory" else "MongoDB connected"}


@app.get("/api/dashboard")
def dashboard(_: dict = Depends(logged_in_user)):
    detections = get_all("detections")
    sessions = get_all("parking_sessions")
    slots = get_all("parking_slots")
    today = now().date()
    today_detections = [d for d in detections if d["timestamp"].date() == today]
    return {
        "totalVehicles": len(get_all("vehicles")),
        "insideParking": len([s for s in sessions if s["status"] == "ACTIVE"]),
        "totalDetections": len(detections),
        "uniqueToday": len({d["plateNumber"] for d in today_detections}),
        "blacklisted": len([d for d in today_detections if d.get("blacklisted")]),
        "activeCameras": len(get_all("cameras")),
        "occupancy": round(100 * len([s for s in slots if s["status"] == "OCCUPIED"]) / len(slots), 1) if slots else 0,
        "recent": sorted(detections, key=lambda d: d["timestamp"], reverse=True)[:8],
    }


@app.get("/api/vehicles")
def vehicles(search: str = "", _: dict = Depends(logged_in_user)):
    items = get_all("vehicles")
    if search:
        items = [v for v in items if search.upper() in v["plateNumber"]]
    return sorted(items, key=lambda v: v["lastSeen"], reverse=True)


@app.get("/api/detections")
def detections(plate: str = "", _: dict = Depends(logged_in_user)):
    items = get_all("detections")
    if plate:
        items = [d for d in items if plate.upper() in d["plateNumber"]]
    return sorted(items, key=lambda d: d["timestamp"], reverse=True)


@app.post("/api/detections")
def save_detection(data: DetectionData, _: dict = Depends(logged_in_user)):
    """Use this route for a real result from the refactored YOLO/EasyOCR pipeline."""
    plate = normal_plate(data.plateNumber)
    record = data.model_dump() | {"plateNumber": plate, "timestamp": now()}
    blacklisted = find_one("blacklist", "plateNumber", plate)
    record["blacklisted"] = bool(blacklisted)

    vehicle = find_one("vehicles", "plateNumber", plate)
    if vehicle:
        update_one("vehicles", "plateNumber", plate, {"lastSeen": record["timestamp"], "visits": vehicle["visits"] + 1})
    else:
        add_one("vehicles", {"plateNumber": plate, "vehicleType": data.vehicleType, "firstSeen": record["timestamp"], "lastSeen": record["timestamp"], "visits": 1})

    saved = add_one("detections", record)
    if blacklisted:
        add_one("alerts", {"plateNumber": plate, "reason": blacklisted["reason"], "time": now(), "message": "BLACKLISTED VEHICLE DETECTED"})
    return saved


@app.post("/api/anpr/image")
async def anpr_image(
    cameraId: str = Form(...),
    file: UploadFile = File(...),
    _: dict = Depends(logged_in_user),
):
    """Run real YOLO + EasyOCR inference. This endpoint never returns fake plates."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(415, "Please upload JPG, JPEG, or PNG image")

    # Models are large files, so they stay beside the program instead of inside it.
    if not VEHICLE_MODEL_PATH.exists() or not PLATE_MODEL_PATH.exists():
        raise HTTPException(503, "Models not found. Put yolov8n.pt and license_plate_detector.pt inside the models folder.")

    try:
        import cv2
        import easyocr
        from ultralytics import YOLO
    except ImportError as error:
        raise HTTPException(503, "Install ultralytics, opencv-python, and easyocr to use real ANPR") from error

    upload_folder = Path("uploads")
    upload_folder.mkdir(exist_ok=True)
    image_path = upload_folder / f"anpr_{int(now().timestamp())}{suffix}"
    image_path.write_bytes(await file.read())
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise HTTPException(400, "The image could not be read")

    vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))
    plate_model = YOLO(str(PLATE_MODEL_PATH))
    reader = easyocr.Reader(["en"], gpu=False)
    vehicle_boxes = []
    for x1, y1, x2, y2, score, class_id in vehicle_model(frame, verbose=False)[0].boxes.data.tolist():
        if int(class_id) in [2, 3, 5, 7]:
            vehicle_boxes.append([x1, y1, x2, y2])

    results = []
    for x1, y1, x2, y2, plate_score, _ in plate_model(frame, verbose=False)[0].boxes.data.tolist():
        # Same simple idea as the original project: a valid plate sits inside a vehicle box.
        if not any(x1 > car[0] and y1 > car[1] and x2 < car[2] and y2 < car[3] for car in vehicle_boxes):
            continue
        crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, clean_crop = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
        text_result = reader.readtext(clean_crop)
        if not text_result:
            continue
        _, raw_plate, ocr_score = text_result[0]
        plate = normal_plate(raw_plate)
        if len(plate) < 4:
            continue
        saved = save_detection(DetectionData(plateNumber=plate, cameraId=cameraId, confidence=round(float(plate_score * ocr_score), 4)), _)
        results.append(saved)
    return {"source": "real YOLOv8 + EasyOCR", "detections": results}


@app.get("/api/cameras")
def cameras(_: dict = Depends(logged_in_user)):
    return get_all("cameras")


@app.post("/api/cameras")
def add_camera(data: CameraData, _: dict = Depends(logged_in_user)):
    if find_one("cameras", "cameraId", data.cameraId):
        raise HTTPException(409, "This camera ID already exists")
    return add_one("cameras", data.model_dump() | {"status": "ONLINE", "createdAt": now()})


@app.get("/api/blacklist")
def blacklist(_: dict = Depends(logged_in_user)):
    return get_all("blacklist")


@app.post("/api/blacklist")
def add_blacklist(data: BlacklistData, _: dict = Depends(logged_in_user)):
    plate = normal_plate(data.plateNumber)
    if find_one("blacklist", "plateNumber", plate):
        raise HTTPException(409, "This plate is already blacklisted")
    return add_one("blacklist", {"plateNumber": plate, "reason": data.reason, "addedAt": now()})


@app.get("/api/parking")
def parking(_: dict = Depends(logged_in_user)):
    return {"slots": get_all("parking_slots"), "sessions": get_all("parking_sessions")}


@app.post("/api/slots")
def add_slot(data: SlotData, _: dict = Depends(logged_in_user)):
    if find_one("parking_slots", "slotNumber", data.slotNumber):
        raise HTTPException(409, "This slot already exists")
    return add_one("parking_slots", {"slotNumber": data.slotNumber, "status": "AVAILABLE", "currentVehicle": None})


@app.post("/api/parking/entry")
def parking_entry(data: ParkingData, _: dict = Depends(logged_in_user)):
    plate = normal_plate(data.plateNumber)
    # A previous closed visit should not stop the same vehicle entering again.
    active_session = next((item for item in get_all("parking_sessions") if item["plateNumber"] == plate and item["status"] == "ACTIVE"), None)
    if active_session:
        raise HTTPException(409, "Vehicle is already inside parking")
    if data.slotNumber:
        slot = find_one("parking_slots", "slotNumber", data.slotNumber)
        if not slot or slot["status"] != "AVAILABLE":
            raise HTTPException(409, "Parking slot is not available")
        update_one("parking_slots", "slotNumber", data.slotNumber, {"status": "OCCUPIED", "currentVehicle": plate})
    return add_one("parking_sessions", {"plateNumber": plate, "cameraId": data.cameraId, "slotNumber": data.slotNumber, "entryTime": now(), "status": "ACTIVE"})


@app.post("/api/parking/exit")
def parking_exit(data: ParkingData, _: dict = Depends(logged_in_user)):
    plate = normal_plate(data.plateNumber)
    session = find_one("parking_sessions", "plateNumber", plate)
    if not session or session["status"] != "ACTIVE":
        raise HTTPException(404, "No active parking session was found")
    minutes = max((now() - session["entryTime"]).total_seconds() / 60, 0)
    # Easy student-project pricing rule. Change these values here whenever needed.
    fee = 30 if minutes <= 60 else 50 if minutes <= 180 else 80 if minutes <= 360 else 120
    update_one("parking_sessions", "plateNumber", plate, {"status": "CLOSED", "exitTime": now(), "fee": fee, "durationMinutes": round(minutes, 1)})
    if session.get("slotNumber"):
        update_one("parking_slots", "slotNumber", session["slotNumber"], {"status": "AVAILABLE", "currentVehicle": None})
    return {"message": "Parking session closed", "fee": fee}


# ----------------------- FRONTEND -----------------------
# Keeping HTML here is not best for a large company app, but it is perfect for
# a student project because there is only one file to run and submit.

PAGE = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartPark AI</title><style>
*{box-sizing:border-box}body{margin:0;background:#071321;color:#e8f0ff;font:14px Arial,sans-serif}.app{display:grid;grid-template-columns:220px 1fr;min-height:100vh}aside{background:#0c1c30;padding:22px}h1{margin:0;color:#4ce1b8}nav button{display:block;width:100%;text-align:left;margin:7px 0;background:transparent;color:#b6c6dc;border:0;padding:11px;border-radius:6px;cursor:pointer}nav button:hover{background:#193858;color:white}main{padding:28px;max-width:1400px;width:100%;margin:auto}.card{background:#10243c;border:1px solid #254465;border-radius:10px;padding:16px;margin:12px 0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat b{font-size:25px;display:block;margin-top:8px}.muted{color:#96abc5}input,select{background:#071a2d;color:white;border:1px solid #345477;border-radius:5px;padding:9px;margin:4px}button{background:#4ce1b8;color:#04271f;border:0;padding:9px 13px;border-radius:5px;font-weight:bold;cursor:pointer}table{border-collapse:collapse;width:100%;margin-top:12px}th,td{padding:10px;text-align:left;border-bottom:1px solid #254465}.login{max-width:380px;margin:12vh auto}.login input{width:100%;margin:6px 0}.slot{display:inline-block;padding:8px;margin:4px;background:#174237;border-radius:5px}.slot.OCCUPIED{background:#54243a}.error{color:#ff9ba7}@media(max-width:700px){.app{grid-template-columns:1fr}aside{padding:12px}nav{display:flex;overflow:auto}nav button{white-space:nowrap}.stats{grid-template-columns:repeat(2,1fr)}main{padding:16px}}
</style></head><body><div id="root"></div><script>
let token=localStorage.smartparkToken||'';
const api=async(path,method='GET',body=null)=>{let r=await fetch('/api'+path,{method,headers:{'Content-Type':'application/json',...(token?{Authorization:'Bearer '+token}:{})},body:body?JSON.stringify(body):null});let d=await r.json();if(!r.ok)throw Error(d.detail||'Something went wrong');return d};
const fmt=t=>t?new Date(t).toLocaleString():'-'; const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function shell(title,content){document.getElementById('root').innerHTML=`<div class="app"><aside><h1>SmartPark AI</h1><p class="muted">Student Project</p><nav>${['Dashboard','Vehicles','Detections','Parking','Blacklist','Cameras'].map(x=>'<button onclick="show(\''+x+'\')">'+x+'</button>').join('')}<button onclick="logout()">Logout</button></nav></aside><main><h2>${title}</h2>${content}</main></div>`}
async function show(page){try{if(page==='Dashboard'){let d=await api('/dashboard');shell('Dashboard',`<div class="stats">${[['Vehicles',d.totalVehicles],['Inside parking',d.insideParking],['Detections',d.totalDetections],['Occupancy',d.occupancy+'%']].map(x=>'<div class="card stat"><span class="muted">'+x[0]+'</span><b>'+x[1]+'</b></div>').join('')}</div><div class="card"><h3>Recent detections</h3>${rows(d.recent,['plateNumber','vehicleType','cameraId','confidence','direction','timestamp'])}</div>`)}
else if(page==='Vehicles'){let d=await api('/vehicles');shell('Vehicles',`<div class="card"><input id="search" placeholder="Search plate"><button onclick="show('Vehicles')">Search</button>${rows(d,['plateNumber','vehicleType','firstSeen','lastSeen','visits'])}</div>`)}
else if(page==='Detections'){let d=await api('/detections');shell('Detection History',`<div class="card">Only save genuine YOLO + OCR results here.${rows(d,['plateNumber','vehicleType','cameraId','confidence','direction','timestamp'])}</div>`)}
else if(page==='Cameras'){let d=await api('/cameras');shell('Cameras',`<div class="card"><input id="camId" placeholder="CAM-01"><input id="camName" placeholder="Main Gate"><input id="camLoc" placeholder="Location"><button onclick="addCamera()">Add camera</button></div><div class="card">${rows(d,['cameraId','name','location','status'])}</div>`)}
else if(page==='Blacklist'){let d=await api('/blacklist');shell('Blacklist',`<div class="card"><input id="blackPlate" placeholder="UP16XY9999"><input id="blackReason" placeholder="Reason"><button onclick="addBlacklist()">Add blacklist plate</button></div><div class="card">${rows(d,['plateNumber','reason','addedAt'])}</div>`)}
else if(page==='Parking'){let d=await api('/parking');shell('Parking Management',`<div class="card"><input id="slot" placeholder="A-01"><button onclick="addSlot()">Add slot</button><div>${d.slots.map(s=>'<span class="slot '+s.status+'">'+esc(s.slotNumber)+' - '+s.status+'</span>').join('')||'No slots added'}</div></div><div class="card"><h3>Entry / Exit</h3><input id="parkPlate" placeholder="Plate number"><input id="parkCamera" placeholder="Camera ID"><input id="parkSlot" placeholder="Slot for entry"><button onclick="parking('entry')">Entry</button><button onclick="parking('exit')">Exit</button>${rows(d.sessions,['plateNumber','slotNumber','entryTime','exitTime','status','fee'])}</div>`)}
}catch(e){alert(e.message)}}
function rows(data,keys){if(!data.length)return '<p class="muted">No data yet.</p>';return '<table><tr>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr>'+data.map(x=>'<tr>'+keys.map(k=>'<td>'+esc(k.includes('Time')||k==='timestamp'||k==='addedAt'||k==='firstSeen'||k==='lastSeen'?fmt(x[k]):x[k])+'</td>').join('')+'</tr>').join('')+'</table>'}
async function addCamera(){await api('/cameras','POST',{cameraId:camId.value,name:camName.value,location:camLoc.value});show('Cameras')}
async function addBlacklist(){await api('/blacklist','POST',{plateNumber:blackPlate.value,reason:blackReason.value});show('Blacklist')}
async function addSlot(){await api('/slots','POST',{slotNumber:slot.value});show('Parking')}
async function parking(type){let b={plateNumber:parkPlate.value,cameraId:parkCamera.value,slotNumber:parkSlot.value||null};let r=await api('/parking/'+type,'POST',b);alert(r.message||('Fee: ₹'+r.fee));show('Parking')}
function logout(){localStorage.removeItem('smartparkToken');token='';loginPage()}
function loginPage(){document.getElementById('root').innerHTML=`<div class="card login"><h1>SmartPark AI</h1><p class="muted">Login: admin / admin123</p><input id="user" value="admin" placeholder="Username"><input id="pass" value="admin123" type="password" placeholder="Password"><button onclick="login()">Login</button><p id="err" class="error"></p></div>`}
async function login(){try{let d=await api('/login','POST',{username:user.value,password:pass.value});token=d.token;localStorage.smartparkToken=token;show('Dashboard')}catch(e){err.textContent=e.message}}
token?show('Dashboard'):loginPage();
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


if __name__ == "__main__":
    import uvicorn
    # reload=False keeps the file simple. In VS Code, stop and run again after edits.
    uvicorn.run(app, host="127.0.0.1", port=8000)

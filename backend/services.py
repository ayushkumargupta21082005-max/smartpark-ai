from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
from .database import utcnow
from .repository import all_docs, find_one, insert_one, update_one


def record_detection(payload: dict) -> dict:
    # The same plate can be seen many times. Keep one vehicle profile and append
    # every sighting to detections so its history is never lost.
    plate = "".join(payload["plateNumber"].upper().split())
    now = payload.get("timestamp") or utcnow()
    payload = {**payload, "plateNumber": plate, "timestamp": now}
    vehicle = find_one("vehicles", {"plateNumber": plate})
    if vehicle:
        update_one("vehicles", {"plateNumber": plate}, {"lastSeen": now, "lastCameraId": payload["cameraId"], "vehicleType": payload["vehicleType"], "detectionCount": vehicle.get("detectionCount", 0) + 1})
    else:
        insert_one("vehicles", {"plateNumber": plate, "vehicleType": payload["vehicleType"], "firstSeen": now, "lastSeen": now, "lastCameraId": payload["cameraId"], "detectionCount": 1, "visits": 0})
    blacklist = find_one("blacklist", {"plateNumber": plate})
    payload["blacklisted"] = bool(blacklist)
    detection = insert_one("detections", payload)
    if blacklist:
        # Alerts are stored separately so the security feed does not have to
        # scan every normal detection.
        insert_one("alerts", {"type": "BLACKLIST", "plateNumber": plate, "cameraId": payload["cameraId"], "timestamp": now, "reason": blacklist["reason"], "detectionId": detection["id"], "read": False})
    return detection


def parking_entry(plate: str, camera_id: str, slot_number: str | None = None) -> dict:
    plate = "".join(plate.upper().split())
    if find_one("parking_sessions", {"plateNumber": plate, "status": "ACTIVE"}):
        raise HTTPException(409, "This vehicle already has an active parking session")
    if slot_number:
        slot = find_one("parking_slots", {"slotNumber": slot_number})
        if not slot or slot["status"] not in ("AVAILABLE", "RESERVED"):
            raise HTTPException(409, "Requested parking slot is unavailable")
        update_one("parking_slots", {"slotNumber": slot_number}, {"status": "OCCUPIED", "currentVehicle": plate})
    vehicle = find_one("vehicles", {"plateNumber": plate})
    if vehicle:
        update_one("vehicles", {"plateNumber": plate}, {"visits": vehicle.get("visits", 0) + 1})
    return insert_one("parking_sessions", {"plateNumber": plate, "cameraEntry": camera_id, "entryTime": utcnow(), "slotNumber": slot_number, "status": "ACTIVE"})


def to_utc_datetime(val) -> datetime:
    if isinstance(val, str):
        val = datetime.fromisoformat(val.replace("Z", "+00:00"))
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    return utcnow()


def parking_exit(plate: str, camera_id: str, rates: list[float]) -> dict:
    plate = "".join(plate.upper().split())
    session = find_one("parking_sessions", {"plateNumber": plate, "status": "ACTIVE"})
    if not session:
        raise HTTPException(404, f"No active parking session for plate '{plate}'")
    exited = utcnow()
    entry_time = to_utc_datetime(session.get("entryTime"))
    hours = max((exited - entry_time).total_seconds() / 3600, 0)
    # Rates come from the environment, not the page, so a pricing change does
    # not require redeploying the frontend.
    fee = rates[0] if hours <= 1 else rates[1] if hours <= 3 else rates[2] if hours <= 6 else rates[3]
    updated = update_one(
        "parking_sessions",
        {"id": session["id"]} if "id" in session else {"plateNumber": plate, "status": "ACTIVE"},
        {
            "status": "CLOSED",
            "exitTime": exited,
            "cameraExit": camera_id,
            "durationMinutes": round(hours * 60, 1),
            "fee": fee,
        }
    )
    if session.get("slotNumber"):
        update_one("parking_slots", {"slotNumber": session["slotNumber"]}, {"status": "AVAILABLE", "currentVehicle": None})
    return updated or {
        **session,
        "status": "CLOSED",
        "exitTime": exited,
        "cameraExit": camera_id,
        "durationMinutes": round(hours * 60, 1),
        "fee": fee,
    }



def dashboard_stats() -> dict:
    now = utcnow()
    today = now.date()
    detections = all_docs("detections")
    sessions = all_docs("parking_sessions")
    slots = all_docs("parking_slots")
    today_detections = [item for item in detections if item["timestamp"].date() == today]
    return {"totalVehicles": len(all_docs("vehicles")), "vehiclesInside": len([item for item in sessions if item["status"] == "ACTIVE"]), "vehiclesExitedToday": len([item for item in sessions if item.get("exitTime") and item["exitTime"].date() == today]), "uniqueVehiclesToday": len({item["plateNumber"] for item in today_detections}), "totalDetections": len(detections), "blacklistedDetected": len([item for item in today_detections if item.get("blacklisted")]), "activeCameras": len([item for item in all_docs("cameras") if item.get("status") == "ONLINE"]), "occupancy": round(100 * len([item for item in slots if item["status"] == "OCCUPIED"]) / len(slots), 1) if slots else 0, "recentDetections": sorted(detections, key=lambda item: item["timestamp"], reverse=True)[:10]}

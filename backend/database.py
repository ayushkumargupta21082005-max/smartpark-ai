"""Database setup. Local development can still open the UI without MongoDB."""
from datetime import datetime, timezone
from typing import Any
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError
from .config import get_settings


class Database:
    def __init__(self) -> None:
        self.client: MongoClient | None = None
        self.db: Any = None
        self.memory: dict[str, list[dict[str, Any]]] = {}
        self.mode = "uninitialized"

    def connect(self) -> None:
        settings = get_settings()
        try:
            self.client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=2500, tz_aware=True)
            self.client.admin.command("ping")
            self.db = self.client[settings.mongo_database]
            self.mode = "mongodb"
            self._indexes()
        except PyMongoError:
            if settings.app_mode != "development":
                raise
            # This is deliberately temporary: it keeps a first-run demo usable,
            # but all data disappears when the server stops.
            self.mode = "memory"
            self.memory = {name: [] for name in ("users", "vehicles", "detections", "parking_sessions", "parking_slots", "cameras", "blacklist", "alerts")}

    def _indexes(self) -> None:
        self.db.vehicles.create_index("plateNumber", unique=True)
        self.db.detections.create_index([("timestamp", DESCENDING)])
        self.db.detections.create_index([("plateNumber", ASCENDING), ("timestamp", DESCENDING)])
        self.db.detections.create_index("cameraId")
        self.db.detections.create_index("direction")
        self.db.blacklist.create_index("plateNumber", unique=True)
        self.db.cameras.create_index("cameraId", unique=True)

    def collection(self, name: str) -> Any:
        if self.mode == "mongodb":
            return self.db[name]
        return self.memory[name]

    def health(self) -> dict[str, str]:
        return {"database": self.mode}


database = Database()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)

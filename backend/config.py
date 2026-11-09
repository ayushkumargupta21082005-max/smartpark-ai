from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_mode: str = "development"
    ai_mode: str = "production"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_database: str = "smartpark_ai"
    jwt_secret: str = "change-this-development-secret"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 480
    upload_dir: str = "uploads"
    ai_model_path: str = "ai/models/yolov8n.pt"
    plate_model_path: str = "ai/models/license_plate_detector.pt"
    confidence_threshold: float = 0.3
    frame_skip: int = 2
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    parking_rates: str = "30,50,80,120"
    max_upload_mb: int = 250
    admin_username: str = "admin"
    admin_password: str = "admin123"

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)

    @property
    def rates(self) -> list[float]:
        return [float(rate) for rate in self.parking_rates.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()

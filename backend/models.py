from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=128)


class CameraCreate(BaseModel):
    cameraId: str = Field(pattern=r"^[A-Za-z0-9_-]{2,32}$")
    name: str = Field(min_length=2, max_length=100)
    location: str = Field(min_length=2, max_length=150)
    streamUrl: str = Field(default="", max_length=1000)
    directionHint: Literal["ENTRY", "EXIT", "UNKNOWN"] = "UNKNOWN"


class BlacklistCreate(BaseModel):
    plateNumber: str = Field(min_length=4, max_length=20)
    reason: str = Field(min_length=3, max_length=250)

    @field_validator("plateNumber")
    @classmethod
    def normalize_plate(cls, value: str) -> str:
        return "".join(value.upper().split())


class ParkingSlotCreate(BaseModel):
    slotNumber: str = Field(min_length=2, max_length=20)
    status: Literal["AVAILABLE", "OCCUPIED", "RESERVED"] = "AVAILABLE"


class ParkingEntry(BaseModel):
    plateNumber: str = Field(min_length=2, max_length=20)
    cameraId: str = "CAM-01"
    slotNumber: str | None = None

    @field_validator("plateNumber")
    @classmethod
    def normalize_plate(cls, value: str) -> str:
        return "".join(value.upper().split())


class ParkingExit(BaseModel):
    plateNumber: str = Field(min_length=2, max_length=20)
    cameraId: str = "CAM-01"

    @field_validator("plateNumber")
    @classmethod
    def normalize_plate(cls, value: str) -> str:
        return "".join(value.upper().split())



class DetectionInput(BaseModel):
    plateNumber: str = Field(min_length=4, max_length=20)
    vehicleType: str = "Unknown"
    cameraId: str
    confidence: float = Field(ge=0, le=1)
    trackingId: int | None = None
    direction: Literal["ENTRY", "EXIT", "UNKNOWN"] = "UNKNOWN"
    imagePath: str | None = None
    timestamp: datetime | None = None

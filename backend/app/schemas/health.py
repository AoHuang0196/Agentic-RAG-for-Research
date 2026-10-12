"""
Health-check response schemas.
"""

from typing import Literal

from pydantic import BaseModel


ServiceStatus = Literal["connected", "not_connected"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    siliconflow: ServiceStatus
    postgres: ServiceStatus
    qdrant: ServiceStatus

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "siliconflow": "connected",
                "postgres": "connected",
                "qdrant": "connected",
            }
        }
    }
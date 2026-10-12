"""
Health-check service.

Performs real connectivity checks against:
  - SiliconFlow    → lightweight model listing call
  - PostgreSQL     → raw asyncpg ping (SELECT 1)
  - Qdrant         → cluster info endpoint
"""

import asyncio

from openai import OpenAI
import asyncpg

from app.config import get_settings
from app.db.qdrant import get_qdrant_client
from app.schemas.health import HealthResponse, ServiceStatus
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def _check_siliconflow(settings) -> ServiceStatus:
    """
    Verify connectivity to the SiliconFlow API.

    SiliconFlow provides an OpenAI-compatible API, so the OpenAI
    Python client can be used for the connectivity check.
    """
    try:
        client = OpenAI(
            api_key=settings.SILICONFLOW_API_KEY,
            base_url=settings.SILICONFLOW_BASE_URL,
        )

        def _list_models():
            return client.models.list()

        models = await asyncio.wait_for(
            asyncio.to_thread(_list_models),
            timeout=10,
        )

        logger.debug(
            "SiliconFlow check OK — %d models available",
            len(models.data),
        )

        return "connected"

    except Exception as exc:
        logger.warning("SiliconFlow check failed: %s", exc)
        return "not_connected"


async def _check_postgres(settings) -> ServiceStatus:
    """Open a raw asyncpg connection and execute SELECT 1."""
    try:
        conn: asyncpg.Connection = await asyncio.wait_for(
            asyncpg.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB,
            ),
            timeout=5,
        )

        await conn.fetchval("SELECT 1")
        await conn.close()

        logger.debug("PostgreSQL check OK")

        return "connected"

    except Exception as exc:
        logger.warning("PostgreSQL check failed: %s", exc)
        return "not_connected"


async def _check_qdrant() -> ServiceStatus:
    """Call Qdrant's cluster info endpoint via the async client."""
    try:
        client = get_qdrant_client()

        info = await asyncio.wait_for(
            client.get_collections(),
            timeout=5,
        )

        logger.debug(
            "Qdrant check OK — %d collection(s)",
            len(info.collections),
        )

        return "connected"

    except Exception as exc:
        logger.warning("Qdrant check failed: %s", exc)
        return "not_connected"


async def get_health() -> HealthResponse:
    """
    Run all connectivity checks concurrently and return an aggregated result.

    Overall status is 'ok' only when all three services are connected.
    """
    settings = get_settings()

    siliconflow_status, postgres_status, qdrant_status = await asyncio.gather(
        _check_siliconflow(settings),
        _check_postgres(settings),
        _check_qdrant(),
    )

    overall = (
        "ok"
        if all(
            s == "connected"
            for s in (
                siliconflow_status,
                postgres_status,
                qdrant_status,
            )
        )
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        siliconflow=siliconflow_status,
        postgres=postgres_status,
        qdrant=qdrant_status,
    )

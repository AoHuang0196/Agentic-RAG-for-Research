"""
Embedding service orchestrator.

Provides a robust, queue-ready retry wrapper around the configured
embedding provider.

The actual embedding implementation is selected through the provider
factory, so this service is independent of any specific model provider.
"""

import asyncio
import random

from app.config import get_settings
from app.services.embeddings import get_embedding_provider
from app.services.metrics import metrics
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_retry_after(exc: Exception) -> float | None:
    """
    Attempt to extract a Retry-After delay from an exception.

    Supports standard HTTP Retry-After headers used by API providers.
    """
    if hasattr(exc, "response") and exc.response is not None:
        headers = getattr(exc.response, "headers", {})
        retry_val = headers.get("retry-after") or headers.get("Retry-After")

        if retry_val is not None:
            try:
                return float(retry_val)
            except (ValueError, TypeError):
                pass

    return None


async def embed_batch_with_retry(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Embed a single batch of texts with robust transient error recovery.

    The configured embedding provider is responsible for communicating
    with the actual embedding API.

    This function handles retries, exponential backoff, jitter,
    and provider rate-limit responses.

    It does NOT handle chunking or large-document batching;
    that remains the caller's responsibility.
    """
    if not texts:
        return []

    settings = get_settings()
    provider = get_embedding_provider()

    retries = 0

    while True:
        metrics.record_attempt()

        try:
            return await asyncio.to_thread(
                provider.embed_batch,
                texts,
                task_type,
            )

        except Exception as exc:
            is_retryable = False
            is_429 = False
            retry_after = None

            # HTTP/API errors from providers such as SiliconFlow.
            status_code = getattr(exc, "status_code", None)

            if status_code in {429, 500, 502, 503, 504}:
                is_retryable = True

                if status_code == 429:
                    is_429 = True
                    metrics.record_429()

                retry_after = _extract_retry_after(exc)

            # Generic network errors.
            elif isinstance(exc, (ConnectionError, TimeoutError, OSError)):
                is_retryable = True

            if not is_retryable or retries >= settings.MAX_EMBED_RETRIES:
                metrics.record_failure(retries)

                if is_retryable:
                    logger.error(
                        "Embedding permanently failed after exhausting "
                        "%d retries.",
                        retries,
                    )

                raise RuntimeError(f"Embedding failed: {exc}") from exc

            # Calculate backoff for next attempt.
            retries += 1

            if retry_after is not None and retry_after > 0:
                sleep_time = retry_after

                logger.warning(
                    "Retry #%d | Waiting %.1fs "
                    "(Provider Retry-After) | Reason: %s",
                    retries,
                    sleep_time,
                    exc.__class__.__name__,
                )

            else:
                sleep_time = min(
                    settings.INITIAL_BACKOFF * (2 ** (retries - 1)),
                    settings.MAX_BACKOFF,
                )

                if settings.ENABLE_JITTER:
                    sleep_time *= random.uniform(0.8, 1.2)

                logger.warning(
                    "Retry #%d | Waiting %.1fs "
                    "(Backoff) | Reason: %s",
                    retries,
                    sleep_time,
                    exc.__class__.__name__,
                )

            await asyncio.sleep(sleep_time)
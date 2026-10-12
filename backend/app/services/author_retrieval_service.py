from __future__ import annotations

import re
from typing import Any

from app.config import get_settings
from app.services.vector_service import get_qdrant_client


def _normalize_text(text: str) -> str:
    """
    Normalize text for robust author-name matching.
    """
    text = text.lower()
    text = text.replace(" ", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_metadata_from_first_chunk(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Extract basic paper metadata from the first Qdrant chunk.

    The current corpus stores title/author/abstract information in the
    first chunk rather than in dedicated metadata fields.
    """
    text = payload.get("text", "") or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Find the Abstract marker.
    abstract_idx = None
    for i, line in enumerate(lines):
        if line.lower() == "abstract":
            abstract_idx = i
            break

    if abstract_idx is None:
        abstract_idx = min(len(lines), 20)

    before_abstract = lines[:abstract_idx]

    # In the current corpus, author information is usually located
    # immediately before "Abstract".
    author_candidates = []
    for line in before_abstract[-5:]:
        normalized = _normalize_text(line)

        # Skip obvious journal/title/DOI/section lines.
        if (
            "doi.org/" in normalized
            or normalized.startswith("http")
            or normalized in {
                "abstract",
                "progress",
                "research article",
                "article",
                "nature",
            }
        ):
            continue

        author_candidates.append(line)

    # The first chunk payload already contains the filename.
    filename = payload.get("filename", "")

    return {
        "document_id": payload.get("document_id"),
        "filename": filename,
        "chunk_index": payload.get("chunk_index"),
        "page_number": payload.get("page_number"),
        "text": text,
        "author_candidates": author_candidates,
    }


async def find_papers_by_author(
    author_name: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Find papers whose first chunk contains the specified author name.

    This version intentionally searches the existing Qdrant corpus directly.
    It does not require a separate metadata database.
    """
    settings = get_settings()
    client = get_qdrant_client()

    target = _normalize_text(author_name)

    results: list[dict[str, Any]] = []
    seen_document_ids: set[str] = set()

    offset = None

    while True:
        points, next_offset = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=None,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            if payload.get("chunk_index") != 0:
                continue

            text = payload.get("text", "") or ""
            normalized_text = _normalize_text(text)

            if target not in normalized_text:
                continue

            document_id = payload.get("document_id")

            if not document_id or document_id in seen_document_ids:
                continue

            seen_document_ids.add(document_id)

            metadata = _extract_metadata_from_first_chunk(payload)

            results.append(
                {
                    "document_id": document_id,
                    "filename": payload.get("filename", ""),
                    "page_number": payload.get("page_number"),
                    "author_query": author_name,
                    "author_match": author_name,
                    "author_candidates": metadata["author_candidates"],
                }
            )

            if limit is not None and len(results) >= limit:
                return results

        if next_offset is None:
            break

        offset = next_offset

    return results

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.services.author_retrieval_service import find_papers_by_author
from app.services.paper_summary_service import summarize_paper


_MAX_CONCURRENT_PAPERS = 3

_CACHE_DIR = (
    Path(__file__).resolve().parents[1]
    / "paper_summaries"
)


def _get_cache_path(document_id: str) -> Path:
    return _CACHE_DIR / f"{document_id}.json"


def _load_cached_summary(document_id: str) -> dict | None:
    cache_path = _get_cache_path(document_id)

    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not data.get("summary"):
            return None

        return data

    except (OSError, json.JSONDecodeError):
        return None


def _save_cached_summary(result: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = _get_cache_path(result["document_id"])

    temp_path = cache_path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    temp_path.replace(cache_path)


async def _summarize_one_paper(
    paper: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    document_id = paper["document_id"]

    # 先检查缓存
    cached = _load_cached_summary(document_id)

    if cached is not None:
        return {
            **cached,
            "filename": paper.get("filename", cached.get("filename", "")),
            "status": "cached",
        }

    # 没有缓存才调用 LLM
    async with semaphore:
        try:
            result = await summarize_paper(
                document_id=document_id,
                batch_size=10,
                overlap=2,
            )

            summary_result = {
                "document_id": document_id,
                "filename": paper.get("filename", ""),
                "summary": result["summary"],
                "chunk_count": result["chunk_count"],
                "batch_count": result["batch_count"],
                "status": "success",
            }

            _save_cached_summary(summary_result)

            return summary_result

        except Exception as exc:
            return {
                "document_id": document_id,
                "filename": paper.get("filename", ""),
                "summary": "",
                "chunk_count": 0,
                "batch_count": 0,
                "status": "error",
                "error": str(exc),
            }


async def summarize_papers_by_author(
    author_name: str,
    max_concurrent_papers: int = _MAX_CONCURRENT_PAPERS,
) -> dict:
    papers = await find_papers_by_author(author_name)

    if not papers:
        return {
            "author": author_name,
            "paper_count": 0,
            "papers": [],
        }

    semaphore = asyncio.Semaphore(max_concurrent_papers)

    tasks = [
        _summarize_one_paper(
            paper=paper,
            semaphore=semaphore,
        )
        for paper in papers
    ]

    results = await asyncio.gather(*tasks)

    return {
        "author": author_name,
        "paper_count": len(results),
        "papers": results,
    }

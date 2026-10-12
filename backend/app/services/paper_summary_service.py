"""
Paper-level summarization service.

Pipeline:
    Qdrant chunks
        ↓
    chunk batching
        ↓
    batch-level summaries (Map)
        ↓
    paper-level synthesis (Reduce)

This service is intentionally independent from the main RAG graph.
It is used for questions that require understanding an entire paper.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.services.vector_service import get_document_chunks
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ── Prompt templates ─────────────────────────────────────────────────────────

_BATCH_SUMMARY_PROMPT = """
You are summarizing a section of a scientific research paper.

Your task is to extract only information that is explicitly supported by the
provided text segment.

Focus on these four aspects:

1. Research objective
   - What scientific question or problem is being investigated?
   - Only include the objective if it is explicitly stated or clearly described
     as the purpose of the current study.

2. Methods
   - What experimental, computational, sequencing, statistical, or analytical
     methods were actually used in the current study?
   - Do not describe methods merely mentioned as background or used by cited
     studies.

3. Main findings/results
   - What results or observations are explicitly reported for the current study?
   - Include quantitative results when they are clearly provided.
   - Do not convert hypotheses, expectations, or background statements into
     findings.

4. Conclusions
   - What conclusions are explicitly supported by the current study?
   - Only include conclusions that can be traced to the reported results.

Important rules:

- Use ONLY the information contained in the provided text segment.
- Do NOT use external knowledge.
- Do NOT infer missing information.
- Do NOT infer information from the paper title.
- Do NOT invent methods, results, conclusions, applications, or biological
  interpretations.
- Do NOT summarize general background knowledge as a finding of the current
  study.
- Do NOT summarize claims from cited or referenced papers as findings of the
  current paper.
- If the text says that "previous studies", "other studies", or specific
  references reported something, treat that information as background rather
  than as a result of the current study.
- Do NOT treat figure captions, table captions, reference lists, acknowledgments,
  author information, or reporting metadata as scientific findings unless the
  text explicitly contains a result relevant to the current study.
- If a category is not supported by the text segment, omit it rather than
  guessing.
- Preserve important quantitative information when explicitly provided.
- Keep the summary concise and factual.

Return the result in the following format:

Research objective:
...

Methods:
...

Main findings:
...

Conclusions:
...

Text segment:
{batch_text}
"""


_REDUCE_SUMMARY_PROMPT = """
You are producing a concise, paper-level summary of a scientific research paper.

The input consists of partial summaries generated from different sections of
the same paper.

Integrate these partial summaries into one coherent summary.

Organize the final summary around:

1. Research objective
2. Methods
3. Main findings/results
4. Conclusions

Strict rules:

- Use ONLY information explicitly present in the partial summaries.
- Do NOT add information from external knowledge.
- Do NOT infer information that is absent from the partial summaries.
- Do NOT introduce new biological interpretations.
- Do NOT introduce new applications or future directions unless they are
  explicitly supported by the partial summaries and are clearly part of the
  paper's conclusions.
- Do NOT treat statements about previous or cited studies as findings of the
  current paper.
- If multiple partial summaries repeat the same information, merge them rather
  than repeating it.
- If partial summaries contain conflicting information, do not resolve the
  conflict using external knowledge; retain only information that is clearly
  supported.
- Prioritize experimentally or computationally demonstrated findings over
  general background statements.
- Quantitative results should be preserved when they are clearly reported.
- Do not mention "partial summaries", "text segments", or the summarization
  process.
- Do not add a separate "Potential applications" or "Future directions"
  section unless explicitly necessary.
- Keep the final summary concise and suitable for understanding the paper as a
  whole.
- Do NOT include introductory phrases such as "Here is the integrated summary"
  or "Following the specified rules". Start directly with "Research objective:".
- Prefer the most important methods rather than listing every technical
  parameter or optimization step.
- Prefer the most important findings rather than enumerating every result.
- The final summary should be concise enough to function as a high-level
  abstract of the paper.

Return the result in the following format:

Research objective:
...

Methods:
...

Main findings:
...

Conclusions:
...

Partial summaries:
{summaries}
"""


# ── LLM factory ──────────────────────────────────────────────────────────────

def _build_llm() -> ChatOpenAI:
    """Build the non-streaming LLM used for paper summarization."""
    settings = get_settings()

    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.SILICONFLOW_API_KEY,
        base_url=settings.SILICONFLOW_BASE_URL,
        temperature=0.1,
        streaming=False,
    )


# ── Chunk batching ───────────────────────────────────────────────────────────

def split_chunks_into_batches(
    chunks: list[dict],
    batch_size: int = 10,
    overlap: int = 2,
) -> list[list[dict]]:
    """
    Split document chunks into overlapping batches.

    Example:
        batch_size=10, overlap=2

        batch 1: chunks 0-9
        batch 2: chunks 8-17
        batch 3: chunks 16-25
        ...

    Args:
        chunks: Qdrant chunk payloads.
        batch_size: Number of chunks in each batch.
        overlap: Number of chunks shared by adjacent batches.

    Returns:
        A list of chunk batches.
    """
    if not chunks:
        return []

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= batch_size:
        raise ValueError("overlap must be smaller than batch_size")

    step = batch_size - overlap

    batches: list[list[dict]] = []

    for start in range(0, len(chunks), step):
        batch = chunks[start:start + batch_size]

        if not batch:
            break

        batches.append(batch)

        if start + batch_size >= len(chunks):
            break

    logger.info(
        "Split %d chunks into %d batches "
        "(batch_size=%d, overlap=%d)",
        len(chunks),
        len(batches),
        batch_size,
        overlap,
    )

    return batches


# ── Batch formatting ─────────────────────────────────────────────────────────

def _format_batch(batch: list[dict]) -> str:
    """
    Convert a chunk batch into a readable text block for the LLM.
    """
    parts: list[str] = []

    for chunk in batch:
        chunk_index = chunk.get("chunk_index", "?")
        page_number = chunk.get("page_number", "?")
        text = chunk.get("text", "")

        if not text:
            continue

        parts.append(
            f"[Chunk {chunk_index}, page {page_number}]\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(parts)


# ── Map ───────────────────────────────────────────────────────────────────────

async def summarize_batch(
    batch: list[dict],
    llm: ChatOpenAI | None = None,
) -> str:
    """
    Summarize one batch of paper chunks.

    This is the Map step of the paper summarization pipeline.
    """
    if not batch:
        return ""

    if llm is None:
        llm = _build_llm()

    context = _format_batch(batch)

    if not context:
        return ""

    prompt = _BATCH_SUMMARY_PROMPT.format(
        batch_text=context,
    )

    response = await llm.ainvoke(
        [HumanMessage(content=prompt)]
    )

    summary = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )

    logger.debug(
        "summarize_batch: chunks %s-%s, summary_chars=%d",
        batch[0].get("chunk_index", "?"),
        batch[-1].get("chunk_index", "?"),
        len(summary),
    )

    return summary.strip()


# ── Reduce ───────────────────────────────────────────────────────────────────

async def synthesize_paper_summary(
    partial_summaries: list[str],
    llm: ChatOpenAI | None = None,
) -> str:
    """
    Combine batch-level summaries into one paper-level summary.

    This is the Reduce step.
    """
    summaries = [
        summary.strip()
        for summary in partial_summaries
        if summary and summary.strip()
    ]

    if not summaries:
        return ""

    if llm is None:
        llm = _build_llm()

    formatted_summaries = "\n\n---\n\n".join(
        f"[Partial summary {i}]\n{summary}"
        for i, summary in enumerate(summaries, start=1)
    )

    prompt = _REDUCE_SUMMARY_PROMPT.format(
        summaries=formatted_summaries,
    )

    response = await llm.ainvoke(
        [HumanMessage(content=prompt)]
    )

    result = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )

    return result.strip()


# ── Public API ───────────────────────────────────────────────────────────────

async def summarize_paper(
    document_id: str,
    batch_size: int = 10,
    overlap: int = 2,
) -> dict:
    """
    Generate a complete summary for one paper.

    Pipeline:
        Qdrant
          ↓
        all chunks
          ↓
        overlapping batches
          ↓
        batch summaries
          ↓
        final paper summary

    Args:
        document_id: Qdrant/PostgreSQL document UUID.
        batch_size: Chunks per Map batch.
        overlap: Shared chunks between adjacent batches.

    Returns:
        Dictionary containing:
            document_id
            filename
            chunk_count
            batch_count
            partial_summaries
            summary
    """
    chunks = await get_document_chunks(document_id)

    if not chunks:
        raise ValueError(
            f"No chunks found for document_id={document_id}"
        )

    batches = split_chunks_into_batches(
        chunks,
        batch_size=batch_size,
        overlap=overlap,
    )

    llm = _build_llm()

    partial_summaries: list[str] = []

    for batch_index, batch in enumerate(batches, start=1):
        logger.info(
            "Summarizing document=%s: batch %d/%d "
            "(chunks %s-%s)",
            document_id,
            batch_index,
            len(batches),
            batch[0].get("chunk_index", "?"),
            batch[-1].get("chunk_index", "?"),
        )

        summary = await summarize_batch(
            batch=batch,
            llm=llm,
        )

        if summary:
            partial_summaries.append(summary)

    if not partial_summaries:
        raise ValueError(
            f"Failed to generate partial summaries "
            f"for document_id={document_id}"
        )

    logger.info(
        "Reducing %d partial summaries for document=%s",
        len(partial_summaries),
        document_id,
    )

    final_summary = await synthesize_paper_summary(
        partial_summaries=partial_summaries,
        llm=llm,
    )

    filename = chunks[0].get("filename", "unknown")

    logger.info(
        "Paper summary completed: document=%s, filename=%s, "
        "chunks=%d, batches=%d, summary_chars=%d",
        document_id,
        filename,
        len(chunks),
        len(batches),
        len(final_summary),
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "batch_count": len(batches),
        "partial_summaries": partial_summaries,
        "summary": final_summary,
    }

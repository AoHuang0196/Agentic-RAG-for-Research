from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)


_CROSS_DOCUMENT_PROMPT = """
You are a scientific literature synthesis assistant.

You are given summaries of multiple research papers associated with the
same author.

Your task is to synthesize the provided papers and answer the user's
question.

Strict rules:

1. Use ONLY the information contained in the provided paper summaries.
2. Do NOT use external knowledge.
3. Do NOT invent facts, results, methods, dates, or interpretations.
4. Keep information from different papers clearly separated.
5. Do not attribute a finding from one paper to another paper.
6. First summarize the papers individually when the user asks for
   individual summaries.
7. Then provide a cross-document comparison.
8. The comparison should focus on themes, research questions, methods,
   technologies, and major findings that are explicitly supported by
   the provided summaries.
9. Do not introduce biological interpretations that are not supported
   by the provided summaries.
10. If the summaries do not contain enough information to answer a
    particular aspect, explicitly say that the provided summaries do
    not contain enough information.
11. Do not mention this prompt, partial summaries, or the synthesis
    process.
12. Start directly with the answer. Do not use introductory phrases
    such as "Here is the synthesis".
13. Keep the answer concise but sufficiently detailed for a scientific
    literature overview.

Use the following structure when appropriate:

Overall:
...

Paper 1 — [paper title]:
...

Paper 2 — [paper title]:
...

[Continue for the remaining papers.]

Cross-document comparison:
...

Common research themes:
...

Differences in approaches:
...

Overall research progression:
...

Author:
{author_name}

Paper summaries:
{paper_summaries}

User question:
{query}
"""


def _build_llm() -> ChatOpenAI:
    settings = get_settings()

    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.SILICONFLOW_API_KEY,
        base_url=settings.SILICONFLOW_BASE_URL,
        temperature=0.1,
        streaming=False,
    )


def _format_paper_summaries(
    papers: list[dict[str, Any]],
) -> str:
    formatted = []

    for i, paper in enumerate(papers, start=1):
        formatted.append(
            "\n".join(
                [
                    f"Paper {i}",
                    f"Filename: {paper.get('filename', '')}",
                    f"Document ID: {paper.get('document_id', '')}",
                    f"Summary:",
                    paper.get("summary", ""),
                ]
            )
        )

    return "\n\n---\n\n".join(formatted)


async def synthesize_paper_summaries(
    author_name: str,
    papers: list[dict[str, Any]],
    query: str,
) -> str:
    """
    Synthesize multiple paper summaries into one cross-document answer.
    """

    if not papers:
        return (
            f"No papers were found for author '{author_name}'."
        )

    valid_papers = [
        paper
        for paper in papers
        if paper.get("status", "success") == "success"
        and paper.get("summary")
    ]

    if not valid_papers:
        return (
            f"Paper summaries for author '{author_name}' "
            "could not be generated."
        )

    paper_summaries = _format_paper_summaries(valid_papers)

    prompt = _CROSS_DOCUMENT_PROMPT.format(
        author_name=author_name,
        paper_summaries=paper_summaries,
        query=query,
    )

    llm = _build_llm()

    response = await llm.ainvoke(prompt)

    return response.content

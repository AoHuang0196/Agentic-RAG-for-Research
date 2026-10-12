from typing import Any

from openai import OpenAI

from app.config import get_settings
from app.services.embeddings.base import BaseEmbeddingProvider


class SiliconFlowEmbeddingProvider(BaseEmbeddingProvider):
    """
    SiliconFlow embedding implementation.

    Uses SiliconFlow's OpenAI-compatible API to generate embeddings.
    The actual embedding model is configured through EMBEDDING_MODEL,
    e.g. BAAI/bge-m3.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

        self.client = OpenAI(
            api_key=self.settings.SILICONFLOW_API_KEY,
            base_url=self.settings.SILICONFLOW_BASE_URL,
        )

        self.model = self.settings.EMBEDDING_MODEL

    def embed_batch(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: Text chunks to embed.
            task_type: Kept for interface compatibility with the previous
                       Gemini implementation. SiliconFlow's embedding API
                       does not use this parameter.

        Returns:
            One embedding vector per input text.
        """
        if not texts:
            return []

        response: Any = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        # SiliconFlow/OpenAI-compatible API returns:
        # response.data = [
        #     EmbeddingObject(embedding=[...], index=0),
        #     ...
        # ]
        #
        # Sort by index to guarantee that the returned vectors correspond
        # to the original input order.
        data = sorted(response.data, key=lambda item: item.index)

        return [list(item.embedding) for item in data]
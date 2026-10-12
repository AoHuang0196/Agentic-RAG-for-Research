from app.services.embeddings.base import BaseEmbeddingProvider
from app.services.embeddings.siliconflow import SiliconFlowEmbeddingProvider


def get_embedding_provider() -> BaseEmbeddingProvider:
    """
    Return the SiliconFlow embedding provider.
    """
    return SiliconFlowEmbeddingProvider()
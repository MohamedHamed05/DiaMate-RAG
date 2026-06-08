import cohere
from .base_provider import RerankingProvider

DEFAULT_RERANKING_MODEL = "rerank-v4.0-pro"
TOP_N = 8

class CohereRerankingProvider(RerankingProvider):
    def __init__(self, api_key: str, default_model: str = DEFAULT_RERANKING_MODEL):
        self.client = cohere.AsyncClientV2(api_key)
        self.default_model = default_model

    async def rerank(self, query: str, documents: list[str], model:str | None = None, top_n: int = TOP_N):
        response = await self.client.rerank(
            model=model or self.default_model,
            query=query,
            documents=documents,
            top_n=top_n
        )

        return response.results


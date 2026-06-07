from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):

    @abstractmethod
    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        pass

    async def get_embedding_size(self) -> int:
        embedding = await self.embed_text("test")
        return len(embedding)
    
class RerankingProvider(ABC):

    @abstractmethod
    async def rerank(self, model:str, query: str, documents: list[str], top_n: int):
        pass


class LLMProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        chat_history: list[dict] | None = None,
        model: str | None = None,
    ) -> str:
        pass

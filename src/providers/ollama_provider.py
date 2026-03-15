import httpx
import math

from .base_provider import EmbeddingProvider, LLMProvider

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "llama3"
DEFAULT_EMBED_BATCH_SIZE = 32


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.batch_size = max(1, batch_size)

    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": model or self.default_model, "input": text},
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["embeddings"][0]

    async def embed_batch(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        all_embeddings = []
        async with httpx.AsyncClient() as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": model or self.default_model, "input": batch},
                    timeout=120.0,
                )
                response.raise_for_status()
                embeddings = response.json()["embeddings"]

                if len(embeddings) != len(batch):
                    raise ValueError("Ollama returned an unexpected number of embeddings")

                for embedding in embeddings:
                    if not all(math.isfinite(x) for x in embedding):
                        raise ValueError("Ollama returned non-finite embedding values")

                all_embeddings.extend(embeddings)

        return all_embeddings


class OllamaLLMProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = DEFAULT_LLM_MODEL):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        chat_history: list[dict] | None = None,
        model: str | None = None,
    ) -> str:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if chat_history:
            messages.extend(chat_history)

        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model or self.default_model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

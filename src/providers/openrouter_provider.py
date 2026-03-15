from openai import AsyncOpenAI

from .base_provider import EmbeddingProvider, LLMProvider

DEFAULT_LLM_MODEL = "meta-llama/llama-3-8b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """OpenRouter does not natively support embeddings.
    Raises NotImplementedError to signal that a different
    embedding provider should be configured."""

    def __init__(self, api_key: str, default_model: str = ""):
        pass

    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        raise NotImplementedError(
            "OpenRouter does not support embeddings. "
            "Use a different EMBEDDING_PROVIDER (google, openai, or ollama)."
        )

    async def embed_batch(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError(
            "OpenRouter does not support embeddings. "
            "Use a different EMBEDDING_PROVIDER (google, openai, or ollama)."
        )


class OpenRouterLLMProvider(LLMProvider):
    def __init__(self, api_key: str, default_model: str = DEFAULT_LLM_MODEL):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
        )
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

        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
        )
        return response.choices[0].message.content

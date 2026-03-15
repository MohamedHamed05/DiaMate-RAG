from openai import AsyncOpenAI

from .base_provider import EmbeddingProvider, LLMProvider

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-5-mini-2025-08-07"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, default_model: str = DEFAULT_EMBEDDING_MODEL):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model

    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        response = await self.client.embeddings.create(
            model=model or self.default_model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=model or self.default_model,
            input=texts,
        )
        return [item.embedding for item in response.data]


class OpenAILLMProvider(LLMProvider):
    def __init__(self, api_key: str, default_model: str = DEFAULT_LLM_MODEL):
        self.client = AsyncOpenAI(api_key=api_key)
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

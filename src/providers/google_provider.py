from google import genai

from .base_provider import EmbeddingProvider, LLMProvider

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_LLM_MODEL = "gemini-2.5-flash"


class GoogleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, default_model: str = DEFAULT_EMBEDDING_MODEL):
        self.client = genai.Client(api_key=api_key)
        self.default_model = default_model

    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        result = self.client.models.embed_content(
            model=model or self.default_model,
            contents=text,
        )
        return result.embeddings[0].values

    async def embed_batch(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = self.client.models.embed_content(
                model=model or self.default_model,
                contents=batch,
            )
            all_embeddings.extend(e.values for e in result.embeddings)
        return all_embeddings


class GoogleLLMProvider(LLMProvider):
    def __init__(self, api_key: str, default_model: str = DEFAULT_LLM_MODEL):
        self.client = genai.Client(api_key=api_key)
        self.default_model = default_model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        chat_history: list[dict] | None = None,
        model: str | None = None,
    ) -> str:
        contents = []

        if chat_history:
            for msg in chat_history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    genai.types.Content(
                        role=role,
                        parts=[genai.types.Part(text=msg["content"])],
                    )
                )

        contents.append(
            genai.types.Content(
                role="user",
                parts=[genai.types.Part(text=prompt)],
            )
        )

        config = None
        if system_prompt:
            config = genai.types.GenerateContentConfig(
                system_instruction=system_prompt
            )

        response = self.client.models.generate_content(
            model=model or self.default_model,
            contents=contents,
            config=config,
        )
        return response.text

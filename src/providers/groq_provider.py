import groq
from .base_provider import LLMProvider

class GroqLLMProvider(LLMProvider):
    def __init__(self, api_key: str, default_model: str = "llama-3.3-70b-versatile"):
        self.client = groq.AsyncGroq(api_key=api_key)
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
            temperature=0.0,
        )
        return response.choices[0].message.content
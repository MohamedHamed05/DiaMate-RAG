from utils.config import Settings

from .base_provider import EmbeddingProvider, LLMProvider
from .google_provider import GoogleEmbeddingProvider, GoogleLLMProvider
from .openai_provider import OpenAIEmbeddingProvider, OpenAILLMProvider
from .openrouter_provider import OpenRouterEmbeddingProvider, OpenRouterLLMProvider
from .ollama_provider import OllamaEmbeddingProvider, OllamaLLMProvider

EMBEDDING_PROVIDERS = {
    "google": lambda s: GoogleEmbeddingProvider(
        api_key=s.GOOGLE_API_KEY,
        default_model=s.EMBEDDING_MODEL or "gemini-embedding-001",
    ),
    "openai": lambda s: OpenAIEmbeddingProvider(
        api_key=s.OPENAI_API_KEY,
        default_model=s.EMBEDDING_MODEL or "text-embedding-3-small",
    ),
    "openrouter": lambda s: OpenRouterEmbeddingProvider(
        api_key=s.OPENROUTER_API_KEY,
    ),
    "ollama": lambda s: OllamaEmbeddingProvider(
        base_url=s.OLLAMA_BASE_URL,
        default_model=s.EMBEDDING_MODEL or "bge-m3:latest",
        batch_size=s.OLLAMA_EMBED_BATCH_SIZE,
    ),
}

LLM_PROVIDERS = {
    "google": lambda s: GoogleLLMProvider(
        api_key=s.GOOGLE_API_KEY,
        default_model=s.LLM_MODEL or "gemini-2.5-flash",
    ),
    "openai": lambda s: OpenAILLMProvider(
        api_key=s.OPENAI_API_KEY,
        default_model=s.LLM_MODEL or "gpt-5-mini-2025-08-07",
    ),
    "openrouter": lambda s: OpenRouterLLMProvider(
        api_key=s.OPENROUTER_API_KEY,
        default_model=s.LLM_MODEL or "openai/gpt-oss-120b",
    ),
    "ollama": lambda s: OllamaLLMProvider(
        base_url=s.OLLAMA_BASE_URL,
        default_model=s.LLM_MODEL or "qwen3.5:9b",
    ),
}


class ProviderFactory:

    @staticmethod
    def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
        provider_name = settings.EMBEDDING_PROVIDER.lower()
        factory_fn = EMBEDDING_PROVIDERS.get(provider_name)
        if not factory_fn:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER: '{provider_name}'. "
                f"Supported: {list(EMBEDDING_PROVIDERS.keys())}"
            )
        return factory_fn(settings)

    @staticmethod
    def create_llm_provider(settings: Settings) -> LLMProvider:
        provider_name = settings.LLM_PROVIDER.lower()
        factory_fn = LLM_PROVIDERS.get(provider_name)
        if not factory_fn:
            raise ValueError(
                f"Unknown LLM_PROVIDER: '{provider_name}'. "
                f"Supported: {list(LLM_PROVIDERS.keys())}"
            )
        return factory_fn(settings)

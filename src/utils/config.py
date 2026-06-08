from pathlib import Path
from pydantic_settings import BaseSettings , SettingsConfigDict

PARENT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    ALLOWED_FILE_TYPES: list
    MAX_FILE_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    # Provider selection
    EMBEDDING_PROVIDER: str = "google"
    LLM_PROVIDER: str = "google"
    RERANKING_PROVIDER: str = ""

    # Provider API keys
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    COHERE_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""

    # Ollama
    OLLAMA_HOST: str = "ollama"
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_EMBED_BATCH_SIZE: int = 32

    # Model overrides (blank = use provider default)
    EMBEDDING_MODEL: str = ""
    LLM_MODEL: str = ""
    RERANKING_MODEL: str = ""

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "diamate"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_CHAT_TTL: int = 86400

    # CORS
    CORS_ORIGINS: list = ["*"]

    model_config = SettingsConfigDict(env_file=PARENT_DIR / '.env')

def get_settings():
    return Settings()
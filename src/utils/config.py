from pathlib import Path
from pydantic_settings import BaseSettings , SettingsConfigDict

PARENT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str    
    GEMINI_API_KEY: str
    ALLOWED_FILE_TYPES: list
    MAX_FILE_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    model_config = SettingsConfigDict(env_file=PARENT_DIR / '.env')

def get_settings():
    return Settings()
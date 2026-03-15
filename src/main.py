from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import base, data
from routes.process import process_router
from routes.chat import chat_router
from utils.config import get_settings
from providers.factory import ProviderFactory
from stores.qdrant_store import QdrantStore
from stores.redis_store import RedisStore
import logging

logger = logging.getLogger('uvicorn.error')


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings

    # Init providers
    embedding_provider = ProviderFactory.create_embedding_provider(settings)
    llm_provider = ProviderFactory.create_llm_provider(settings)
    app.state.embedding_provider = embedding_provider
    app.state.llm_provider = llm_provider

    # Init Qdrant
    qdrant_store = QdrantStore(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        collection_name=settings.QDRANT_COLLECTION,
    )
    app.state.qdrant_store = qdrant_store

    # Auto-create collection if not exists
    current_model = settings.EMBEDDING_MODEL or "default"
    if not qdrant_store.collection_exists():
        embedding_size = await embedding_provider.get_embedding_size()
        qdrant_store.create_collection(
            embedding_size=embedding_size,
            embedding_model=current_model,
        )
        logger.info(f"Created Qdrant collection '{settings.QDRANT_COLLECTION}' "
                     f"with dimension {embedding_size}")
        app.state.embedding_model_mismatch = False
    else:
        stored_model = qdrant_store.get_collection_model()
        if stored_model and stored_model != current_model:
            logger.warning(
                f"Embedding model mismatch: stored='{stored_model}', "
                f"configured='{current_model}'. "
                f"Call POST /api/v1/process/reindex to re-embed."
            )
            app.state.embedding_model_mismatch = True
        else:
            app.state.embedding_model_mismatch = False

    # Init Redis
    redis_store = RedisStore(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        ttl=settings.REDIS_CHAT_TTL,
    )
    app.state.redis_store = redis_store

    yield

    logger.info("Shutting down...")

    if hasattr(app.state, 'qdrant_store') and app.state.qdrant_store:
        try:
            app.state.qdrant_store.client.close()
            logger.info("Qdrant client closed.")
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")

    if hasattr(app.state, 'redis_store') and app.state.redis_store.connected:
        try:
            app.state.redis_store.client.close()
            logger.info("Redis client closed.")
        except Exception as e:
            logger.warning(f"Error closing Redis client: {e}")

    logger.info("Shutdown complete.")


tags_metadata = [
    {"name": "base", "description": "Health check and app info"},
    {"name": "data", "description": "Upload, list, and delete document files"},
    {"name": "process", "description": "Embed documents into Qdrant and check embedding status"},
    {"name": "chat", "description": "RAG chat, session management, and conversation history"},
]

app = FastAPI(
    title="DiaMate-RAG API",
    description="Retrieval-Augmented Generation system for diabetes-related health information",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(process_router)
app.include_router(chat_router)

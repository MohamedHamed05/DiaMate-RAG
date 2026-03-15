from providers.base_provider import EmbeddingProvider, LLMProvider
from stores.qdrant_store import QdrantStore
from stores.redis_store import RedisStore
import logging

logger = logging.getLogger('uvicorn.error')

SYSTEM_PROMPT = (
    "You are a diabetes health assistant. "
    "Answer ONLY based on the provided context below. "
    "If the context doesn't contain enough information to answer the question, "
    "say that you don't have enough information. "
    "Do not make up or infer medical information.\n\n"
    "Context:\n{context}"
)

TOP_K = 5


class ChatController:
    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        qdrant_store: QdrantStore,
        redis_store: RedisStore,
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.qdrant_store = qdrant_store
        self.redis_store = redis_store

    async def answer(self, session_id: str, question: str) -> dict:
        query_vector = await self.embedding_provider.embed_text(question)

        search_results = self.qdrant_store.search(
            query_vector=query_vector,
            top_k=TOP_K,
        )

        context_parts = [result["text"] for result in search_results]
        context = "\n---\n".join(context_parts) if context_parts else "No relevant context found."

        system_prompt = SYSTEM_PROMPT.format(context=context)

        chat_history = self.redis_store.get_history(session_id)

        response = await self.llm_provider.generate(
            prompt=question,
            system_prompt=system_prompt,
            chat_history=chat_history,
        )

        self.redis_store.add_message(session_id, "user", question)
        self.redis_store.add_message(session_id, "assistant", response)

        source_chunks = [
            {
                "text": r["text"],
                "file_id": r["file_id"],
                "score": r["score"],
                "metadata": r.get("metadata", {}),
            }
            for r in search_results
        ]

        return {
            "answer": response,
            "source_chunks": source_chunks,
        }

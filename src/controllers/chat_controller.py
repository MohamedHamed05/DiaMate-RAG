from providers.base_provider import EmbeddingProvider, LLMProvider, RerankingProvider
from stores.qdrant_store import QdrantStore
from stores.redis_store import RedisStore
import logging

logger = logging.getLogger('uvicorn.error')

SYSTEM_PROMPT = (
    "You are a diabetes health assistant. "
    "Answer ONLY based on the provided context below. "
    "If the context doesn't contain enough information to answer the question, "
    "say that you don't have enough information. "
    "Do not make up or infer medical information beyond what is stated.\n\n"
    "When the context contains relevant information, provide a THOROUGH and "
    "COMPLETE answer. Include all relevant details, specific values, "
    "qualifications, exceptions, and clinical nuances mentioned in the text. "
    "Do not stop at a partial answer when additional relevant details are "
    "available in the context.\n\n"
    "Include related consequences, implications, or qualifications that the "
    "text connects to the main topic, even if they are not directly asked about.\n\n"
    "When the context contains specific numerical data, statistics, "
    "percentages, risk ratios, or threshold values, include them in your "
    "answer rather than using general descriptions. Precise numbers from the "
    "source text are always preferred over vague qualifiers like 'higher risk' "
    "or 'substantial increase' when the exact figures are available.\n\n"
    "Match the language and register of the user's question. "
    "If the question is in Egyptian Arabic (colloquial/عامية), respond in Egyptian Arabic. "
    "If the question is in Modern Standard Arabic (فصحى), respond in MSA. "
    "If the question is in English, respond in English."
    "Context:\n{context}"
)

TOP_K = 5
TOP_K_RERANK = 40
TOP_N_RERANK = 8


class ChatController:
    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        qdrant_store: QdrantStore,
        redis_store: RedisStore,
        reranking_provider: RerankingProvider = None,
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.qdrant_store = qdrant_store
        self.redis_store = redis_store
        self.reranking_provider = reranking_provider

    async def answer(self, session_id: str, question: str) -> dict:
        query_vector = await self.embedding_provider.embed_text(question)

        if self.reranking_provider:
            search_results = self.qdrant_store.search(query_vector=query_vector, top_k=TOP_K_RERANK)
            if search_results:
                reranked = await self.reranking_provider.rerank(
                    query=question,
                    documents=[r["text"] for r in search_results],
                    top_n=TOP_N_RERANK,
                )
                search_results = [
                    {**search_results[r.index], "score": r.relevance_score}
                    for r in reranked
                ]
            else:
                search_results = search_results[:TOP_K]
        else:
            search_results = self.qdrant_store.search(query_vector=query_vector, top_k=TOP_K)

        context = "\n---\n".join(r["text"] for r in search_results) or "No relevant context found."
        system_prompt = SYSTEM_PROMPT.format(context=context)
        chat_history = self.redis_store.get_history(session_id)

        response = await self.llm_provider.generate(
            prompt=question,
            system_prompt=system_prompt,
            chat_history=chat_history,
        )

        self.redis_store.add_message(session_id, "user", question)
        self.redis_store.add_message(session_id, "assistant", response)

        return {
            "answer": response,
            "source_chunks": [
                {"text": r["text"], "file_id": r["file_id"], "score": r["score"], "metadata": r.get("metadata", {})}
                for r in search_results
            ],
        }

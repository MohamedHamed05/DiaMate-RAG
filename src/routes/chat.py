from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from .schemes.chat_scheme import (
    ChatRequest, ChatResponse, SessionClearedResponse,
    SourceChunk, ChatHistoryResponse, SessionListResponse,
)
from controllers.chat_controller import ChatController
from models import ResponseSignal
import logging

logger = logging.getLogger('uvicorn.error')

chat_router = APIRouter(
    prefix='/api/v1/chat',
    tags=['chat']
)


@chat_router.post('/', response_model=ChatResponse, summary="Send a chat question")
async def chat(request: Request, body: ChatRequest):
    if getattr(request.app.state, "embedding_model_mismatch", False):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={'Signal': ResponseSignal.EMBEDDING_MODEL_MISMATCH.value}
        )

    chat_controller = ChatController(
        llm_provider=request.app.state.llm_provider,
        embedding_provider=request.app.state.embedding_provider,
        qdrant_store=request.app.state.qdrant_store,
        redis_store=request.app.state.redis_store,
    )

    try:
        result = await chat_controller.answer(
            session_id=body.session_id,
            question=body.question,
        )
        return ChatResponse(
            Signal=ResponseSignal.CHAT_SUCCESS.value,
            answer=result['answer'],
            source_chunks=[
                SourceChunk(**chunk) for chunk in result['source_chunks']
            ],
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'Signal': ResponseSignal.CHAT_FAIL.value}
        )


@chat_router.get('/sessions', response_model=SessionListResponse, summary="List active sessions")
async def list_sessions(request: Request):
    redis_store = request.app.state.redis_store
    sessions = redis_store.get_sessions()
    return SessionListResponse(
        Signal=ResponseSignal.CHAT_SUCCESS.value,
        sessions=sessions,
    )


@chat_router.get('/{session_id}', response_model=ChatHistoryResponse, summary="Get chat history")
async def get_history(request: Request, session_id: str):
    redis_store = request.app.state.redis_store
    messages = redis_store.get_history(session_id)
    return ChatHistoryResponse(
        Signal=ResponseSignal.CHAT_SUCCESS.value,
        session_id=session_id,
        messages=messages,
    )


@chat_router.delete('/{session_id}', response_model=SessionClearedResponse, summary="Clear a session")
async def clear_session(request: Request, session_id: str):
    redis_store = request.app.state.redis_store
    redis_store.clear_history(session_id)
    return SessionClearedResponse(Signal='Session Cleared Successfully')

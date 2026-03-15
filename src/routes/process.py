from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from .schemes.process_scheme import (
    EmbedFileRequest, EmbedAllRequest,
    EmbedResponse, EmbedAllResponse, ReindexResponse,
    EmbeddingStatusResponse,
)
from controllers import ProcessController
from models import ResponseSignal
import logging

logger = logging.getLogger('uvicorn.error')

process_router = APIRouter(
    prefix='/api/v1/process',
    tags=['process']
)


def _check_model_mismatch(request: Request):
    return getattr(request.app.state, "embedding_model_mismatch", False)


@process_router.get('/status', response_model=EmbeddingStatusResponse, summary="Embedding status")
async def embedding_status(request: Request):
    process_controller = ProcessController(
        qdrant_store=request.app.state.qdrant_store,
    )
    all_file_ids = process_controller.get_file_ids()
    stored_file_ids = request.app.state.qdrant_store.get_stored_file_ids()
    pending = [f for f in all_file_ids if f not in stored_file_ids]

    return EmbeddingStatusResponse(
        Signal=ResponseSignal.FILE_PROCESS_SUCCESS.value,
        total_files=len(all_file_ids),
        embedded_files=len(stored_file_ids),
        pending_files=len(pending),
        pending_file_ids=pending,
        model_mismatch=_check_model_mismatch(request),
    )


@process_router.post('/push', response_model=EmbedResponse, summary="Embed a single file")
async def push_file(request: Request, body: EmbedFileRequest):
    if _check_model_mismatch(request):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={'Signal': ResponseSignal.EMBEDDING_MODEL_MISMATCH.value}
        )

    process_controller = ProcessController(
        embedding_provider=request.app.state.embedding_provider,
        qdrant_store=request.app.state.qdrant_store,
    )
    file_ids = process_controller.get_file_ids()

    if body.file_id not in file_ids:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'Signal': ResponseSignal.FILE_NOT_FOUND.value}
        )

    stored_file_ids = request.app.state.qdrant_store.get_stored_file_ids()
    if body.file_id in stored_file_ids:
        return EmbedResponse(
            Signal=ResponseSignal.EMBEDDING_PUSH_SUCCESS.value,
            inserted_count=0,
            message='File already embedded, skipped.',
        )

    try:
        count = await process_controller.embed_file_by_id(
            file_id=body.file_id,
            chunk_size=body.chunk_size,
            overlap_size=body.overlap_size,
        )
        return EmbedResponse(
            Signal=ResponseSignal.EMBEDDING_PUSH_SUCCESS.value,
            inserted_count=count,
        )
    except Exception as e:
        logger.error(f"Error pushing embeddings: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'Signal': ResponseSignal.EMBEDDING_PUSH_FAIL.value}
        )


@process_router.post('/push_all', response_model=EmbedAllResponse, summary="Embed all new files")
async def push_all(request: Request, body: EmbedAllRequest):
    if _check_model_mismatch(request):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={'Signal': ResponseSignal.EMBEDDING_MODEL_MISMATCH.value}
        )

    process_controller = ProcessController(
        embedding_provider=request.app.state.embedding_provider,
        qdrant_store=request.app.state.qdrant_store,
    )

    try:
        files_count, inserted_count = await process_controller.embed_new_files(
            chunk_size=body.chunk_size,
            overlap_size=body.overlap_size,
        )
        return EmbedAllResponse(
            Signal=ResponseSignal.EMBEDDING_PUSH_SUCCESS.value,
            new_files_processed=files_count,
            inserted_count=inserted_count,
        )
    except Exception as e:
        logger.error(f"Error pushing all embeddings: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'Signal': ResponseSignal.EMBEDDING_PUSH_FAIL.value}
        )


@process_router.post('/reindex', response_model=ReindexResponse, summary="Reindex all files")
async def reindex(request: Request, body: EmbedAllRequest):
    process_controller = ProcessController(
        embedding_provider=request.app.state.embedding_provider,
        qdrant_store=request.app.state.qdrant_store,
    )

    settings = request.app.state.settings
    embedding_model = settings.EMBEDDING_MODEL or "default"

    try:
        files_count, inserted_count = await process_controller.reindex_all(
            embedding_model=embedding_model,
            chunk_size=body.chunk_size,
            overlap_size=body.overlap_size,
        )
        request.app.state.embedding_model_mismatch = False
        return ReindexResponse(
            Signal=ResponseSignal.EMBEDDING_REINDEX_SUCCESS.value,
            files_processed=files_count,
            inserted_count=inserted_count,
        )
    except Exception as e:
        logger.error(f"Error during reindex: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'Signal': ResponseSignal.EMBEDDING_PUSH_FAIL.value}
        )

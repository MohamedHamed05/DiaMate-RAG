from fastapi import APIRouter, Request, UploadFile, Depends, status
from fastapi.responses import JSONResponse
from controllers import DataController
from utils.config import Settings, get_settings
from .schemes.data_scheme import UploadFileResponse, ListFilesResponse, DeleteFileResponse
from models import ResponseSignal
import logging
import aiofiles

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix='/api/v1/data',
    tags=['data']
)


@data_router.get('/files', response_model=ListFilesResponse, summary="List uploaded files")
async def list_files():
    data_controller = DataController()
    file_ids = data_controller.get_file_ids()
    return ListFilesResponse(
        Signal=ResponseSignal.FILE_PROCESS_SUCCESS.value,
        files=file_ids,
    )


@data_router.delete('/{file_id}', response_model=DeleteFileResponse,
                    summary="Delete a file and its embeddings",
                    description="Removes the file from disk and deletes all its associated chunks from Qdrant.")
async def delete_file(request: Request, file_id: str):
    data_controller = DataController()
    deleted = data_controller.delete_file(file_id)
    if not deleted:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={'Signal': ResponseSignal.FILE_DELETE_FAILED.value}
        )
    request.app.state.qdrant_store.delete_by_file_id(file_id)
    return DeleteFileResponse(Signal=ResponseSignal.FILE_DELETE_SUCCESS.value)


@data_router.post('/upload', response_model=UploadFileResponse, summary="Upload a document file")
async def upload_file(file: UploadFile,
                      app_settings: Settings = Depends(get_settings)):
    data_controller = DataController()
    isValid, signal = data_controller.validate_file(file=file)
    if not isValid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'Signal': signal}
        )
    
    file_base_dir = data_controller.get_file_dir()
    unique_filename = data_controller.generate_unique_filename(filename=file.filename)
    file_dir = file_base_dir / unique_filename

    try:
        async with aiofiles.open(file_dir, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        logger.error(f"Error while uploading file: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                'Signal': ResponseSignal.FILE_UPLOAD_FAILED.value,
            }
        )

    return UploadFileResponse(
        Signal=ResponseSignal.FILE_UPLOAD_SUCCESS.value,
        File_ID=unique_filename,
    )

from fastapi import APIRouter, File, UploadFile, Depends, status
from fastapi.responses import JSONResponse
from controllers import DataController, ProcessController
from utils.config import Settings, get_settings
from .schemes.data_scheme import ProcessRequest
from models import ResponseSignal
import logging
import aiofiles
import os

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix='/api/v1/data',
    tags=['data']
)

@data_router.post('/upload')
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
            content= {
                'Signal': ResponseSignal.FILE_UPLOAD_FAILED.value,

            }
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'Signal': ResponseSignal.FILE_UPLOAD_SUCCESS.value,
            'File_ID': unique_filename
        }
    )

@data_router.post('/process')
async def process(request: ProcessRequest):

    file_id = request.file_id
    chunk_size = request.chunk_size
    overlap_size = request.overlap_size
    start_over = request.startover

    project_controller = ProcessController()

    file_content = project_controller.get_content(file_id)

    chunks = project_controller.process_file_content(file_content=file_content,
                                                     file_id=file_id,
                                                     chunk_size=chunk_size,
                                                     overlap_size=overlap_size)
    
    if chunks is None or len(chunks) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                'Signal': ResponseSignal.FILE_PROCCESS_FAIL.value   
            }
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'signal': ResponseSignal.FILE_PROCCESS_SUCCESS.value,
            'Ex. Chunk': chunks[0].page_content
        }
    )





    

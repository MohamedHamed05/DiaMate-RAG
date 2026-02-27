from fastapi import APIRouter, File, UploadFile, Depends, status
from fastapi.responses import JSONResponse
from controllers import DataController
from utils.config import Settings, get_settings
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
            'Signal': ResponseSignal.FILE_UPLOAD_SUCCESS.value
        }
    )
    

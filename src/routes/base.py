from fastapi import APIRouter, UploadFile, Depends, status
from fastapi.responses import JSONResponse
from utils.config import Settings, get_settings
import os

base_router = APIRouter(
    prefix="/api/v1"
)

@base_router.get("/")
async def health(app_settings: Settings=Depends(get_settings)):
    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'App Name': app_name,
            'App Version': app_version,
            'Status':'Good'
        }
    )
  
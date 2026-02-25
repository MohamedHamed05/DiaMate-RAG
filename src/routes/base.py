from fastapi import APIRouter
import os

base_router = APIRouter()
@base_router.get("/")
def health():
    app_name = os.getenv('APP_NAME')
    app_version = os.getenv('APP_VERSION')
    return {
        'App': f'{app_name}_{app_version}',
        'Status':'Good'
    }
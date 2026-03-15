from pydantic import BaseModel


class UploadFileResponse(BaseModel):
    Signal: str
    File_ID: str = ""


class ListFilesResponse(BaseModel):
    Signal: str
    files: list[str] = []


class DeleteFileResponse(BaseModel):
    Signal: str
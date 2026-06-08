from pydantic import BaseModel, Field
from typing import Optional


class EmbedFileRequest(BaseModel):
    file_id: str = Field(..., min_length=1)
    chunk_size: Optional[int] = Field(default=400, gt=0)
    overlap_size: Optional[int] = Field(default=0, ge=0)

    model_config = {"json_schema_extra": {
        "examples": [{"file_id": "abc123_diabetes.txt", "chunk_size": 400, "overlap_size": 20}]
    }}


class EmbedAllRequest(BaseModel):
    chunk_size: Optional[int] = Field(default=400, gt=0)
    overlap_size: Optional[int] = Field(default=0, ge=0)


class EmbedResponse(BaseModel):
    Signal: str
    inserted_count: int = 0
    message: str = ""


class EmbedAllResponse(BaseModel):
    Signal: str
    new_files_processed: int = 0
    inserted_count: int = 0


class ReindexResponse(BaseModel):
    Signal: str
    files_processed: int = 0
    inserted_count: int = 0


class EmbeddingStatusResponse(BaseModel):
    Signal: str
    total_files: int = 0
    embedded_files: int = 0
    pending_files: int = 0
    pending_file_ids: list[str] = []
    model_mismatch: bool = False

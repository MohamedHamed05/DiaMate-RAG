from pydantic import BaseModel
from typing import Optional

class ProcessFileRequest(BaseModel):
    file_id: str
    chunk_size: Optional[int] = 400
    overlap_size: Optional[int] = 0

class ProcessAllRequest(BaseModel):
    chunk_size: Optional[int] = 400
    overlap_size: Optional[int] = 0
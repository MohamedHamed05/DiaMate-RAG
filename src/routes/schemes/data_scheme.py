from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    file_id: str
    chunk_size: Optional[int] = 400
    overlap_size: Optional[int] = 0
    startover: Optional[bool] = False
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)

    model_config = {"json_schema_extra": {
        "examples": [{"session_id": "user-123", "question": "What are the symptoms of type 2 diabetes?"}]
    }}


class SourceChunk(BaseModel):
    text: str
    file_id: str
    score: float
    metadata: dict = {}


class ChatResponse(BaseModel):
    Signal: str
    answer: str
    source_chunks: list[SourceChunk] = []


class SessionClearedResponse(BaseModel):
    Signal: str


class ChatHistoryResponse(BaseModel):
    Signal: str
    session_id: str
    messages: list[dict] = []


class SessionListResponse(BaseModel):
    Signal: str
    sessions: list[str] = []

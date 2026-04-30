from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class IngestMessagesRequest(BaseModel):
    batch_size: int = Field(default=1000, ge=1)
    recreate: bool = False


class IngestThreadsRequest(BaseModel):
    recreate: bool = False


class SearchThreadsRequest(BaseModel):
    project_hint: str
    limit: int = Field(default=10, ge=1, le=100)


class CorpusBatchRequest(BaseModel):
    project_hint: str
    offset: int = Field(default=0, ge=0)
    batch_size: int = Field(default=50, ge=1, le=500)
    thread_limit: int = Field(default=30, ge=1, le=200)


class RunBatchAgentRequest(BaseModel):
    project_hint: str
    model: str = "deepseek-chat"


class RunGlobalAgentRequest(BaseModel):
    project_hint: str
    model: str = "deepseek-chat"
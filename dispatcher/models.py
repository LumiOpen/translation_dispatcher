from pydantic import BaseModel
from typing import Optional

class WorkItem(BaseModel):
    row_id: int
    row_content: str

class ResultSubmission(BaseModel):
    row_id: int
    result: str

class Status(BaseModel):
    last_processed_row: int
    next_row_id: int
    issued: int
    pending: int
    heap_size: int
    expired_reissues: int

class WorkResponse(BaseModel):
    status: str  # "OK", "all_work_complete", or "retry"
    retry_in: Optional[int] = None  # seconds to wait if status is "retry"
    work: Optional[WorkItem] = None


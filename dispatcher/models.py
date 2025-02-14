from pydantic import BaseModel
from typing import Optional
from enum import Enum

class WorkStatus(str, Enum):
    OK = "OK"
    ALL_WORK_COMPLETE = "all_work_complete"
    RETRY = "retry"
    SERVER_UNAVAILABLE = "server_unavailable"

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
    status: WorkStatus
    retry_in: Optional[int] = None  # seconds to wait if status is "retry"
    work: Optional[WorkItem] = None

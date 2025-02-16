import argparse
import uvicorn
import logging
import threading
import time
import sys
from fastapi import FastAPI, Query
from dispatcher.models import (
    WorkItem,
    BatchWorkResponse,
    BatchResultSubmission,
    BatchResultResponse,
    WorkStatus,
)
from dispatcher.data_tracker import DataTracker  # your existing logic, adapted for batch usage

app = FastAPI()

dt = None            # Global DataTracker
retry_time = 300     # default wait time
shutdown_interval = 5

def background_shutdown():
    while True:
        time.sleep(shutdown_interval)
        global dt
        if dt is not None and dt.all_work_complete():
            logging.info("All work complete. Shutting down server.")
            sys.exit(0)

@app.on_event("startup")
def startup_event():
    global dt
    if dt is None:
        raise RuntimeError("DataTracker is not initialized!")
    # start a background thread to check for completion
    threading.Thread(target=background_shutdown, daemon=True).start()

@app.get("/work", response_model=BatchWorkResponse)
def get_work(batch_size: int = Query(1, ge=1)):
    global dt, retry_time
    if dt.all_work_complete():
        return BatchWorkResponse(status=WorkStatus.ALL_WORK_COMPLETE, items=[])

    items = []
    for _ in range(batch_size):
        row = dt.get_next_row()
        if row is None:
            # No more input lines
            break
        work_id, content = row
        items.append(WorkItem(work_id=work_id, content=content))

    if not items:
        # We read no new lines
        if dt.all_work_complete():
            return BatchWorkResponse(status=WorkStatus.ALL_WORK_COMPLETE, items=[])
        else:
            # The input is exhausted but pending work not done => "retry"
            return BatchWorkResponse(status=WorkStatus.RETRY, retry_in=retry_time)

    # otherwise, success
    return BatchWorkResponse(status=WorkStatus.OK, items=items)

@app.post("/results", response_model=BatchResultResponse)
def submit_results(batch: BatchResultSubmission):
    """
    Accept a list of WorkItem objects. For each item that includes work_id and result,
    call DataTracker.complete_row(work_id, result).
    """
    global dt
    success_count = 0
    for wi in batch.items:
        if wi.result is not None:
            dt.complete_row(wi.work_id, wi.result)
            success_count += 1
        else:
            logging.warning(f"WorkItem work_id={wi.work_id} had no result, ignoring.")
    return BatchResultResponse(status=WorkStatus.OK, count=success_count)

@app.get("/status")
def get_status():
    global dt
    return {
        "last_processed_row": dt.last_processed_row,
        "next_work_id": dt.next_row_id,
        "issued": len(dt.issued),
        "pending": len(dt.pending_write),
        "heap_size": len(dt.issued_heap),
        "expired_reissues": dt.expired_reissues
    }

def main():
    parser = argparse.ArgumentParser(description="Dispatcher Server")
    parser.add_argument("--infile", type=str, required=True, help="Input file path")
    parser.add_argument("--outfile", type=str, required=True, help="Output file path")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file path")
    parser.add_argument("--retry", type=int, default=300, help="Retry time in seconds")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    args = parser.parse_args()

    global dt, retry_time
    retry_time = args.retry
    checkpoint_path = args.checkpoint if args.checkpoint else args.outfile + ".checkpoint"
    dt = DataTracker(args.infile, args.outfile, checkpoint_path)
    logging.info("Server starting with infile=%s, outfile=%s, checkpoint=%s, retry_time=%d",
                 args.infile, args.outfile, checkpoint_path, retry_time)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )

if __name__ == "__main__":
    main()

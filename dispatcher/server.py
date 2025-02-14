import argparse
import uvicorn
import logging
import threading
import time
import sys
from fastapi import FastAPI, HTTPException
from dispatcher.models import WorkItem, ResultSubmission, WorkResponse, Status
from dispatcher.data_tracker import DataTracker

app = FastAPI()

# Global DataTracker instance and global retry time.
dt = None
retry_time = 300  # default retry time in seconds

def background_shutdown():
    """Background thread: if all work is complete, shut down the server."""
    while True:
        time.sleep(5)  # check every 5 seconds
        global dt
        if dt is not None and dt.all_work_complete():
            logging.info("All work complete. Shutting down server.")
            # Gracefully shutdown the server.
            sys.exit(0)

@app.on_event("startup")
def startup_event():
    global dt
    if dt is None:
        logging.error("DataTracker is not initialized!")
        raise Exception("DataTracker is not initialized")
    # Start the background thread to check for shutdown.
    threading.Thread(target=background_shutdown, daemon=True).start()

@app.get("/work", response_model=WorkResponse)
def get_work():
    global dt, retry_time
    if dt.all_work_complete():
        return WorkResponse(status="all_work_complete")
    work = dt.get_next_row()
    if work is None:
        # Input file exhausted but pending work exists → ask client to retry.
        return WorkResponse(status="retry", retry_in=retry_time)
    row_id, row_content = work
    return WorkResponse(status="OK", work=WorkItem(row_id=row_id, row_content=row_content))

@app.post("/result")
def submit_result(submission: ResultSubmission):
    global dt
    dt.complete_row(submission.row_id, submission.result)
    return {"status": "success", "row_id": submission.row_id}

@app.get("/status", response_model=Status)
def get_status():
    global dt
    return Status(
        last_processed_row=dt.last_processed_row,
        next_row_id=dt.next_row_id,
        issued=len(dt.issued),
        pending=len(dt.pending_write),
        heap_size=len(dt.issued_heap),
        expired_reissues=dt.expired_reissues
    )

def main():
    parser = argparse.ArgumentParser(description="Dispatcher Server")
    parser.add_argument("--infile", type=str, required=True, help="Input file path")
    parser.add_argument("--outfile", type=str, required=True, help="Output file path")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file path")
    parser.add_argument("--retry", type=int, default=300, help="Retry time in seconds (default: 300)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    args = parser.parse_args()

    global retry_time, dt
    retry_time = args.retry
    checkpoint_path = args.checkpoint if args.checkpoint else args.outfile + ".checkpoint"
    dt = DataTracker(args.infile, args.outfile, checkpoint_path)
    logging.info(f"Server starting with infile={args.infile}, outfile={args.outfile}, checkpoint={checkpoint_path}, retry_time={retry_time}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

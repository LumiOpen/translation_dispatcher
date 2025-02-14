# Dispatcher

Simple library to dispatch work from a large line-oriented file (jsonl) for
distributed workers without pre-apportioning work.

Dispatcher is ideal for batch inference workloads where individual requests
may take varying amounts of time, but you want to keep all workers busy and
avoid the long tails that you might run into by dividing the work up
beforehand.

Dispatcher guarantees that each completed work item will be persisted to disk
only once, but items may be processed more than once, so it is inappropriate
for work that changes state in external systems or is otherwise not idempotent.

Work is checkpointed so that if a job ends unexpectedly, work can begin where
it left off with minimal lost work (specifically, only work which is cached
waiting to be written because it has been completed out of order will be lost.)

In order to work efficiently with large data files, ensure each item is written
only once, and avoid costly scans and reconciliation on restart, the
dispatcher works on a line-per-line basis, each nth line of input will
correspond with the nth line of output. On restarting, we only need to
determine where we left off to begin again.

This means the dispatcher must cache out of order work until the work can be
written contiguously in the output file. Work that has been issued but not
completed will be reissued again after a timeout to avoid unbounded memory
growth, but in certain pathological situations (a "query of death") this could
still cause an out of memory situation.

Probably we should time out incomplete work after a certain number of retries
and write it to a rejected file, but that is not yet implemented.


## To Develop

```bash
pip install -e .[dev]
```

## To run the server
```bash
python -m dispatcher.server --infile path/to/input.jsonl --outfile path/to/output.jsonl
# or
dispatcher-server --infile path/to/input.jsonl --outfile path/to/output.jsonl
```

## Client example
```python
import time
from dispatcher.client import WorkClient
from dispatcher.models import WorkStatus

client = WorkClient("http://127.0.0.1:8000")

while True:
    work_resp = client.get_work()
    
    if work_resp.status == WorkStatus.ALL_WORK_COMPLETE:
        print("All work complete. Exiting.")
        break
        
    elif work_resp.status == WorkStatus.RETRY:
        print(f"No work available; retry in {work_resp.retry_in} seconds.")
        time.sleep(work_resp.retry_in)
        continue
        
    elif work_resp.status == WorkStatus.SERVER_UNAVAILABLE:
        # The server is not running.
        # The server exits once all work is complete, so let's assume that's the case here.
        print("Server is unavailable. Exiting.")
        break
        
    elif work_resp.status == WorkStatus.OK and work_resp.work:
        work_item = work_resp.work
        print(f"Got work: row_id={work_item.row_id}, content='{work_item.row_content}'")
        # Process the work (replace with actual processing).
        result = f"processed_{work_item.row_content}"
        submit_resp = client.submit_result(work_item.row_id, result)
        print(f"Submitted result for row {work_item.row_id}: {submit_resp}")

```

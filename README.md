### Dispatcher

Tool to reliably dispatch work from a large line-oriented file (jsonl) for
distributed workers.

# To Develop

```bash
pip install -e .[dev]
```

# To run the server
```bash
python -m dispatcher.server --infile path/to/input.jsonl --outfile path/to/output.jsonl
# or
dispatcher-server --infile path/to/input.jsonl --outfile path/to/output.jsonl
```

# Client example
```python
from dispatcher.client import WorkClient
import time

client = WorkClient("http://127.0.0.1:8000")

while True:
    work_resp = client.get_work()
    if work_resp.status == "all_work_complete":
        print("All work complete. Exiting.")
        break
    elif work_resp.status == "retry":
        print(f"Retry in {work_resp.retry_in} seconds.")
        time.sleep(work_resp.retry_in)
        continue
    elif work_resp.status == "OK" and work_resp.work:
        work_item = work_resp.work
        print(f"Got work: row_id={work_item.row_id}, content='{work_item.row_content}'")
        result = f"processed_{work_item.row_content}"
        submit_resp = client.submit_result(work_item.row_id, result)
        print(f"Submitted result for row {work_item.row_id}: {submit_resp}")
```

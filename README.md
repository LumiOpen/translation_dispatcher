# Dispatcher



```bash
python -m dispatcher.server --infile path/to/input.jsonl --outfile path/to/output.jsonl
```

From clients:
```python
import time
from dispatcher.client import WorkClient

client = WorkClient("http://127.0.0.1:8000")

while True:
    work_resp = client.get_work()
    if work_resp.status == "all_work_complete":
        print("All work complete. Exiting.")
        break
    elif work_resp.status == "retry":
        print(f"No work available; retry in {work_resp.retry_in} seconds.")
        time.sleep(work_resp.retry_in)
        continue
    work_item = work_resp.work
    print(f"Got work: row_id={work_item.row_id}, content='{work_item.row_content}'")
    # Process work (replace with real logic)
    result = f"processed_{work_item.row_content}"
    submit_resp = client.submit_result(work_item.row_id, result)
    print(f"Submitted result for row {work_item.row_id}: {submit_resp}")
```

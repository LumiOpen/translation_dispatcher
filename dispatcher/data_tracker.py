import os
import time
import json
import logging
import heapq

logging.basicConfig(level=logging.INFO)

class DataTracker:
    def __init__(self, infile_path, outfile_path, checkpoint_path,
                 work_timeout=300, checkpoint_interval=60):
        """
        Parameters:
          - infile_path: Path to the input JSONL file.
          - outfile_path: Path to the output JSONL file.
          - checkpoint_path: Path to the checkpoint file.
          - work_timeout: Seconds after which issued work is considered expired.
          - checkpoint_interval: Seconds between checkpoint writes.
        """
        self.infile_path = infile_path
        self.outfile_path = outfile_path
        self.checkpoint_path = checkpoint_path
        self.work_timeout = work_timeout
        self.checkpoint_interval = checkpoint_interval

        self.last_processed_row = -1   # Last contiguous row written.
        self.next_row_id = 0           # Next row id to assign.

        self.input_offset = 0
        self.output_offset = 0

        self.last_checkpoint_time = time.time()
        self.expired_reissues = 0

        self.issued = {}            # row_id -> row_content
        self.issued_heap = []       # min-heap of (timestamp, row_id)
        self.pending = {}           # row_id -> result

        self.infile = open(self.infile_path, "r")
        self.outfile = open(self.outfile_path, "a+")
        self._load_checkpoint()

    def _load_checkpoint(self):
        # If a checkpoint file exists and is non-empty, load its state.
        if os.path.exists(self.checkpoint_path) and os.path.getsize(self.checkpoint_path) > 0:
            try:
                with open(self.checkpoint_path, "r") as f:
                    cp = json.load(f)
            except json.JSONDecodeError:
                cp = {}
            self.last_processed_row = cp.get("last_processed_row", -1)
            self.input_offset = cp.get("input_offset", 0)
            self.output_offset = cp.get("output_offset", 0)
            self.infile.seek(self.input_offset)
            self.outfile.seek(self.output_offset)
            # Reconcile extra output lines:
            extra_lines = self.outfile.readlines()
            extra_count = len(extra_lines)
            # For each extra line in the output, discard one line from the input.
            for _ in range(extra_count):
                self.infile.readline()
            self.input_offset = self.infile.tell()
            self.last_processed_row += extra_count
            self.next_row_id = self.last_processed_row + 1
            self.outfile.seek(0, os.SEEK_END)
            self.output_offset = self.outfile.tell()
        else:
            self.last_processed_row = -1
            self.input_offset = 0
            self.output_offset = 0
            self.next_row_id = 0

    def all_work_complete(self) -> bool:
        """
        Returns True if the input file is exhausted and no pending work remains.
        """
        remaining = os.stat(self.infile_path).st_size - self.infile.tell()
        return remaining == 0 and len(self.pending) == 0

    def get_next_row(self):
        now = time.time()
        while self.issued_heap:
            heap_ts, row_id = self.issued_heap[0]
            if row_id not in self.issued:
                heapq.heappop(self.issued_heap)
                continue
            if now - heap_ts > self.work_timeout:
                heapq.heappop(self.issued_heap)
                new_ts = now
                heapq.heappush(self.issued_heap, (new_ts, row_id))
                self.expired_reissues += 1
                return row_id, self.issued[row_id]
            else:
                break

        line = self.infile.readline()
        if not line:
            return None  # End of file.
        row_content = line.rstrip("\n")
        row_id = self.next_row_id
        self.next_row_id += 1
        ts = now
        self.issued[row_id] = row_content
        heapq.heappush(self.issued_heap, (ts, row_id))
        return row_id, row_content

    def complete_row(self, row_id, result):
        if row_id <= self.last_processed_row or row_id in self.pending:
            logging.warning(f"Duplicate completion for row {row_id}; discarding.")
            return
        if row_id not in self.issued:
            logging.warning(f"Completion for row {row_id} not issued; discarding.")
            return
        del self.issued[row_id]
        if row_id == self.last_processed_row + 1:
            self._write_result(row_id, result)
            self.last_processed_row = row_id
            self._flush_pending()
        else:
            self.pending[row_id] = result

        now = time.time()
        if now - self.last_checkpoint_time >= self.checkpoint_interval:
            self._write_checkpoint()
            self.last_checkpoint_time = now
            logging.info(f"Checkpoint: last_processed_row={self.last_processed_row}, "
                         f"input_offset={self.infile.tell()}, output_offset={self.outfile.tell()}, "
                         f"issued={len(self.issued)}, pending={len(self.pending)}, "
                         f"heap_size={len(self.issued_heap)}, expired_reissues={self.expired_reissues}")

    def _flush_pending(self):
        next_expected = self.last_processed_row + 1
        while next_expected in self.pending:
            result = self.pending.pop(next_expected)
            self._write_result(next_expected, result)
            self.last_processed_row = next_expected
            next_expected = self.last_processed_row + 1

    def _write_result(self, row_id, result):
        self.outfile.write(result + "\n")
        self.outfile.flush()
        self.output_offset = self.outfile.tell()

    def _write_checkpoint(self):
        cp = {
            "last_processed_row": self.last_processed_row,
            "input_offset": self.infile.tell(),
            "output_offset": self.outfile.tell()
        }
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(cp, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(temp_path, self.checkpoint_path)

    def close(self):
        self._write_checkpoint()
        self.infile.close()
        self.outfile.close()

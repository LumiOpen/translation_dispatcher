import os
import time
import json
import logging
import heapq
import threading

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
        self.issued_heap = []       # min-heap of (timestamp, row_id); uses lazy deletion
        self.pending_write = {}     # row_id -> result

        self._state_lock = threading.Lock()

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

            # Any lines after the output_offset in in output_file have been
            # completed after the checkpoint is written, so we need to move
            # past them in both the outfile and the infile
            extra_lines = self.outfile.readlines()
            extra_count = len(extra_lines)
            self.output_offset = self.outfile.tell()

            # For each extra line in the output, discard one line from the input.
            for _ in range(extra_count):
                self.infile.readline()
            self.input_offset = self.infile.tell()

            self.last_processed_row += extra_count
            self.next_row_id = self.last_processed_row + 1

            logging.info(f"Loaded checkpoint: last_processed_row={self.last_processed_row}, "
                         f"input_offset={self.input_offset}, output_offset={self.output_offset}")
        else:
            self.last_processed_row = -1
            self.input_offset = 0
            self.output_offset = 0
            self.next_row_id = 0
            logging.info("No checkpoint found; starting fresh.")

    def all_work_complete(self) -> bool:
        """
        Returns True if the input file is exhausted and no pending work remains.
        """
        remaining = os.stat(self.infile_path).st_size - self.infile.tell()
        return remaining == 0 and len(self.pending_write) == 0

    def get_next_row(self):
        with self._state_lock:
            now = time.time()
            # check first for expired work needing to be reissued
            while self.issued_heap:
                heap_ts, row_id = self.issued_heap[0]
                # lazy deleteion
                if row_id not in self.issued:
                    heapq.heappop(self.issued_heap)
                    continue
                # see if the oldest iten in minheap needs to be reissued
                if now - heap_ts > self.work_timeout:
                    heapq.heappop(self.issued_heap)
                    return self._issue_work(now, self.issued[row_id], row_id)
                break

            # get new work
            line = self.infile.readline()
            if not line:
                return None  # End of file.
            row_content = line.rstrip("\n")

            return self._issue_work(now, row_content)

    def complete_row(self, row_id, result):
        with self._state_lock:
            if row_id <= self.last_processed_row or row_id in self.pending_write:
                logging.warning(f"Duplicate completion for row {row_id}; discarding.")
                return
            if row_id not in self.issued:
                logging.warning(f"Completion for row {row_id} not issued; discarding.")
                return
            del self.issued[row_id]
            if self._can_write(row_id):
                self._write_result(row_id, result)
                self._flush_pending()
            else:
                self.pending_write[row_id] = result

            now = time.time()
            if now - self.last_checkpoint_time >= self.checkpoint_interval:
                self._write_checkpoint()
                self.last_checkpoint_time = now
                logging.info(f"Checkpoint: last_processed_row={self.last_processed_row}, "
                             f"input_offset={self.infile.tell()}, output_offset={self.outfile.tell()}, "
                             f"issued={len(self.issued)}, pending={len(self.pending_write)}, "
                             f"heap_size={len(self.issued_heap)}, expired_reissues={self.expired_reissues}")

    def _issue_work(self, when, row_content, row_id=None):
        if row_id is None:
            row_id = self.next_row_id
            self.next_row_id += 1
            self.issued[row_id] = row_content
        else:
            # this is reissued work
            self.expired_reissues += 1
            logging.info("Reissuing {row_id} after expiration ({self.expired_reissues=}).")
            assert(row_id in self.issued)
        heapq.heappush(self.issued_heap, (when, row_id))
        return row_id, row_content

    def _can_write(self, row_id):
        return row_id == self.last_processed_row + 1

    def _flush_pending(self):
        next_expected = self.last_processed_row + 1
        while next_expected in self.pending_write:
            result = self.pending_write.pop(next_expected)
            self._write_result(next_expected, result)
            next_expected += 1

    def _write_result(self, row_id, result):
        self.outfile.write(result + "\n")
        self.outfile.flush()
        self.output_offset = self.outfile.tell()
        self.last_processed_row = row_id


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
        with self._state_lock:
            # Write a final checkpoint and log status before shutting down.
            self._write_checkpoint()
            logging.info(f"Final checkpoint written: last_processed_row={self.last_processed_row}, "
                         f"input_offset={self.infile.tell()}, output_offset={self.outfile.tell()}, "
                         f"issued={len(self.issued)}, pending={len(self.pending_write)}, "
                         f"heap_size={len(self.issued_heap)}, expired_reissues={self.expired_reissues}")
            self.infile.close()
            self.outfile.close()

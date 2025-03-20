import os
import time
import json
import tempfile
import unittest
from dispatcher.data_tracker import DataTracker

# Use short timeouts for testing.
WORK_TIMEOUT = 2      # seconds
CHECKPOINT_INTERVAL = 1  # seconds

class TestDataTracker(unittest.TestCase):
    def setUp(self):
        # Create temporary files for input and output.
        self.infile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        self.outfile = tempfile.NamedTemporaryFile(mode="a+", delete=False)
        # Use mktemp for the checkpoint file.
        self.checkpoint = tempfile.mktemp()
        self.infile.close()
        self.outfile.close()

        # Write sample rows to the input file (7 rows).
        with open(self.infile.name, "w") as f:
            for i in range(7):
                f.write(f"row_content_{i}\n")

        # Ensure checkpoint file does not exist (simulate cold start).
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

    def tearDown(self):
        os.remove(self.infile.name)
        os.remove(self.outfile.name)
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

    def test_cold_start(self):
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        self.assertEqual(dt.last_processed_work_id, -1)
        self.assertEqual(dt.input_offset, 0)
        self.assertEqual(dt.next_work_id, 0)
        dt.close()

    def test_get_work_batch_and_reissue(self):
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        row, = dt.get_work_batch()
        self.assertIsNotNone(row)
        row_id, content = row
        self.assertEqual(content, "row_content_0")
        time.sleep(WORK_TIMEOUT + 0.5)
        reissued, = dt.get_work_batch()
        self.assertEqual(reissued[0], row_id)
        self.assertEqual(reissued[1], content)
        row2, = dt.get_work_batch()
        self.assertEqual(row2[0], row_id + 1)
        self.assertEqual(row2[1], "row_content_1")
        dt.close()

    def test_complete_in_order_and_out_of_order(self):
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()  # row 0
        r1, = dt.get_work_batch()  # row 1
        r2, = dt.get_work_batch()  # row 2

        dt.complete_work_batch([(r0[0], "result_0")])
        with open(self.outfile.name, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].strip(), "result_0")

        dt.complete_work_batch([(r2[0], "result_2")])
        self.assertEqual(dt.last_processed_work_id, 0)
        self.assertIn(r2[0], dt.pending_write)

        dt.complete_work_batch([(r1[0], "result_1")])
        self.assertEqual(dt.last_processed_work_id, 2)
        with open(self.outfile.name, "r") as f:
            lines = f.readlines()
        self.assertEqual([line.strip() for line in lines],
                         ["result_0", "result_1", "result_2"])
        dt.close()

    def test_duplicate_completion(self):
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()  # row 0
        dt.complete_work_batch([(r0[0], "result_0")])
        dt.complete_work_batch([(r0[0], "result_duplicate")])
        with open(self.outfile.name, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].strip(), "result_0")
        dt.close()

    def test_checkpoint_written(self):
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt.get_work_batch()
        dt.complete_work_batch([(r0[0], "result_0")])
        r1, = dt.get_work_batch()
        dt.complete_work_batch([(r1[0], "result_1")])
        time.sleep(CHECKPOINT_INTERVAL + 0.5)
        r2, = dt.get_work_batch()
        dt.complete_work_batch([(r2[0], "result_2")])
        with open(self.checkpoint, "r") as f:
            cp = json.load(f)
        self.assertEqual(cp.get("last_processed_work_id"), 2)
        dt.close()

    def test_load_from_checkpoint(self):
        # Process 2 rows and write a checkpoint.
        dt1 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt1.get_work_batch()  # row 0
        dt1.complete_work_batch([(r0[0], "result_0")])
        r1, = dt1.get_work_batch()  # row 1
        dt1.complete_work_batch([(r1[0], "result_1")])
        dt1._write_checkpoint()
        dt1.close()
        with open(self.checkpoint, "r") as f:
            cp = json.load(f)
        self.assertEqual(cp.get("last_processed_work_id"), 1)

        # Create a new DataTracker from the same files.
        dt2 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        self.assertEqual(dt2.last_processed_work_id, 1)
        self.assertEqual(dt2.next_work_id, 2)
        r2, = dt2.get_work_batch()
        self.assertEqual(r2[0], 2)
        self.assertEqual(r2[1], "row_content_2")
        dt2.complete_work_batch([(r2[0], "result_2")])
        with open(self.outfile.name, "r") as f:
            lines = f.readlines()
        self.assertEqual([line.strip() for line in lines],
                         ["result_0", "result_1", "result_2"])
        dt2.close()

    def test_load_from_checkpoint_with_extra_rows(self):
        """
        Process some rows, then process additional rows after the checkpoint was written.
        Upon loading, the DataTracker should:
          - Seek to the saved output offset,
          - For each extra output line, read and discard one input line,
          - Update last_processed_work_id and next_work_id accordingly.
        """
        dt1 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt1.get_work_batch()  # row 0
        dt1.complete_work_batch([(r0[0], "result_0")])
        r1, = dt1.get_work_batch()  # row 1
        dt1.complete_work_batch([(r1[0], "result_1")])
        r2, = dt1.get_work_batch()  # row 2
        dt1.complete_work_batch([(r2[0], "result_2")])
        # Write a checkpoint now; it records last_processed_work_id==2.
        dt1._write_checkpoint()
        # Now process additional rows.
        r3, = dt1.get_work_batch()  # row 3
        dt1.complete_work_batch([(r3[0], "result_3")])
        r4, = dt1.get_work_batch()  # row 4
        dt1.complete_work_batch([(r4[0], "result_4")])

        # checkpoint should reflect the earlier state
        with open(self.checkpoint, "r") as f:
            cp = json.load(f)
        self.assertEqual(cp.get("last_processed_work_id"), 2)
        
        # Now load a new tracker and ensure it reconciles correctly.
        dt2 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        self.assertEqual(dt2.last_processed_work_id, 4)
        self.assertEqual(dt2.next_work_id, 5)
        r5, = dt2.get_work_batch()
        self.assertEqual(r5[0], 5)
        self.assertEqual(r5[1], "row_content_5")
        dt2.close()

    def test_load_from_checkpoint_with_extra_rows_unwritten(self):
        """
        Process some rows, then process additional rows after the checkpoint was written.
        Upon loading, the DataTracker should:
          - Seek to the saved output offset,
          - For each extra output line, read and discard one input line,
          - Update last_processed_work_id and next_work_id accordingly.
        """
        dt1 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        r0, = dt1.get_work_batch()  # row 0
        dt1.complete_work_batch([(r0[0], "result_0")])
        r1, = dt1.get_work_batch()  # row 1
        dt1.complete_work_batch([(r1[0], "result_1")])
        r2, = dt1.get_work_batch()  # row 2
        dt1.complete_work_batch([(r2[0], "result_2")])
        # Write a checkpoint now; it records last_processed_work_id==2.
        dt1._write_checkpoint()
        # Now process additional rows.
        r3, = dt1.get_work_batch()  # row 3
        r4, = dt1.get_work_batch()  # row 4

        # checkpoint should reflect the earlier state
        with open(self.checkpoint, "r") as f:
            cp = json.load(f)
        self.assertEqual(cp.get("last_processed_work_id"), 2)

        # Now load a new tracker and it continues from the last written record
        dt2 = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                          work_timeout=WORK_TIMEOUT, checkpoint_interval=CHECKPOINT_INTERVAL)
        self.assertEqual(dt2.last_processed_work_id, 2)
        self.assertEqual(dt2.next_work_id, 3)
        r5, = dt2.get_work_batch()
        self.assertEqual(r5[0], 3)
        self.assertEqual(r5[1], "row_content_3")
        dt2.close()

if __name__ == "__main__":
    unittest.main()

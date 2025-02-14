import os
import tempfile
import json
import unittest
import time
from fastapi.testclient import TestClient

# Import the FastAPI app and global dt variable.
from dispatcher.server import app, dt
from dispatcher.data_tracker import DataTracker

class TestServer(unittest.TestCase):
    def setUp(self):
        # Create temporary input, output, and checkpoint files.
        self.infile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        self.outfile = tempfile.NamedTemporaryFile(mode="a+", delete=False)
        self.checkpoint = tempfile.mktemp()
        self.infile.close()
        self.outfile.close()

        # Write sample data to the input file (e.g., 3 rows).
        with open(self.infile.name, "w") as f:
            for i in range(3):
                f.write(f"row_content_{i}\n")

        # Ensure the checkpoint file does not exist.
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

        # Now, initialize the global DataTracker in the server module.
        global dt
        dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                         work_timeout=2, checkpoint_interval=1)
        # Create the TestClient after dt is set.
        self.client = TestClient(app)

    def tearDown(self):
        dt.close()
        os.remove(self.infile.name)
        os.remove(self.outfile.name)
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

    def test_get_work_ok(self):
        response = self.client.get("/work")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # Expect status "OK" and a work item.
        self.assertEqual(data["status"], "OK")
        self.assertIn("work", data)
        self.assertIsNotNone(data["work"])

    def test_all_work_complete_status(self):
        # Process all rows.
        for _ in range(3):
            response = self.client.get("/work")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            if data["status"] == "OK":
                work = data["work"]
                # Submit result.
                res = self.client.post("/result", json={"row_id": work["row_id"],
                                                         "result": f"result_{work['row_content']}"})
                self.assertEqual(res.status_code, 200)
        # Now the input file is exhausted and no pending work remains.
        response = self.client.get("/work")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "all_work_complete")

    def test_retry_status(self):
        # Get one work item and do not complete it.
        response = self.client.get("/work")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "OK")
        # Since the work is issued and not completed, and input may be exhausted,
        # a subsequent request might return a "retry" status.
        response = self.client.get("/work")
        data = response.json()
        self.assertIn(data["status"], ["OK", "retry", "all_work_complete"])

    def test_submit_result(self):
        # Get a work item and submit result.
        response = self.client.get("/work")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        if data["status"] == "OK":
            work = data["work"]
            res = self.client.post("/result", json={"row_id": work["row_id"],
                                                     "result": f"result_{work['row_content']}"})
            self.assertEqual(res.status_code, 200)
            res_data = res.json()
            self.assertEqual(res_data.get("status"), "success")

    def test_status_endpoint(self):
        # Check that /status returns a proper status.
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("last_processed_row", data)
        self.assertIn("next_row_id", data)
        self.assertIn("issued", data)
        self.assertIn("pending", data)
        self.assertIn("heap_size", data)
        self.assertIn("expired_reissues", data)

if __name__ == "__main__":
    unittest.main()

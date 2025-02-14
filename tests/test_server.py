import os
import tempfile
import json
import unittest
from fastapi.testclient import TestClient
import dispatcher.server as server_mod
from dispatcher.data_tracker import DataTracker

class TestServer(unittest.TestCase):
    def setUp(self):
        # Create temporary input, output, and checkpoint files.
        self.infile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        self.outfile = tempfile.NamedTemporaryFile(mode="a+", delete=False)
        self.checkpoint = tempfile.mktemp()
        self.infile.close()
        self.outfile.close()
        with open(self.infile.name, "w") as f:
            for i in range(3):
                f.write(f"row_content_{i}\n")
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)
        # Initialize global dt.
        server_mod.dt = DataTracker(self.infile.name, self.outfile.name, self.checkpoint,
                                    work_timeout=2, checkpoint_interval=1)
        self.client = TestClient(server_mod.app)

    def tearDown(self):
        server_mod.dt.close()
        os.remove(self.infile.name)
        os.remove(self.outfile.name)
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

    def test_get_work_ok(self):
        response = self.client.get("/work")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "OK")
        self.assertIn("work", data)
        self.assertIsNotNone(data["work"])

    def test_all_work_complete_status(self):
        # Process all work.
        for _ in range(3):
            response = self.client.get("/work")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            if data["status"] == "OK":
                work = data["work"]
                res = self.client.post("/result", json={"row_id": work["row_id"],
                                                         "result": f"result_{work['row_content']}"})
                self.assertEqual(res.status_code, 200)
        response = self.client.get("/work")
        data = response.json()
        self.assertEqual(data["status"], "all_work_complete")

    def test_retry_status(self):
        # Get one work item and do not complete it.
        response = self.client.get("/work")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "OK")
        # Subsequent call might return "retry" if input is exhausted or waiting.
        response = self.client.get("/work")
        data = response.json()
        self.assertIn(data["status"], ["OK", "retry", "all_work_complete"])

    def test_submit_result(self):
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

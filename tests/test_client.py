import unittest
import responses
from dispatcher.client import WorkClient
from dispatcher.models import WorkResponse, WorkItem, Status, ResultSubmission, WorkStatus

class TestWorkClient(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://testserver"
        self.client = WorkClient(self.base_url)

    @responses.activate
    def test_get_work_ok(self):
        # Simulate a GET /work returning a valid work item.
        work_data = {
            "status": "OK",
            "work": {"row_id": 1, "row_content": "test_content"}
        }
        responses.add(
            responses.GET,
            f"{self.base_url}/work",
            json=work_data,
            status=200
        )
        work = self.client.get_work()
        self.assertEqual(work.status, WorkStatus.OK)
        self.assertIsNotNone(work.work)
        self.assertEqual(work.work.row_id, 1)
        self.assertEqual(work.work.row_content, "test_content")

    @responses.activate
    def test_get_work_all_complete(self):
        # Simulate a GET /work returning 404 (no more work).
        responses.add(
            responses.GET,
            f"{self.base_url}/work",
            json={"detail": "No more work available"},
            status=404
        )
        work = self.client.get_work()
        self.assertEqual(work.status, WorkStatus.ALL_WORK_COMPLETE)

    @responses.activate
    def test_get_work_retry(self):
        # Simulate a GET /work returning a retry status.
        work_data = {
            "status": "retry",
            "retry_in": 10,
            "work": None
        }
        responses.add(
            responses.GET,
            f"{self.base_url}/work",
            json=work_data,
            status=200
        )
        work = self.client.get_work()
        self.assertEqual(work.status, WorkStatus.RETRY)
        self.assertEqual(work.retry_in, 10)
        self.assertIsNone(work.work)

    @responses.activate
    def test_submit_result(self):
        # Simulate a POST /result returning success.
        result_data = {"status": "success", "row_id": 1}
        responses.add(
            responses.POST,
            f"{self.base_url}/result",
            json=result_data,
            status=200
        )
        resp = self.client.submit_result(1, "processed_test")
        self.assertEqual(resp["status"], "success")
        self.assertEqual(resp["row_id"], 1)

    @responses.activate
    def test_get_status(self):
        # Simulate a GET /status response.
        status_data = {
            "last_processed_row": 5,
            "next_row_id": 6,
            "issued": 0,
            "pending": 0,
            "heap_size": 0,
            "expired_reissues": 2
        }
        responses.add(
            responses.GET,
            f"{self.base_url}/status",
            json=status_data,
            status=200
        )
        status_obj = self.client.get_status()
        self.assertEqual(status_obj.last_processed_row, 5)
        self.assertEqual(status_obj.next_row_id, 6)
        self.assertEqual(status_obj.issued, 0)
        self.assertEqual(status_obj.pending, 0)
        self.assertEqual(status_obj.heap_size, 0)
        self.assertEqual(status_obj.expired_reissues, 2)

if __name__ == "__main__":
    unittest.main()

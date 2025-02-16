import unittest
import responses
import requests
from dispatcher.client import WorkClient
from dispatcher.models import (
    WorkItem, 
    WorkStatus,
    BatchWorkResponse, 
    BatchResultResponse
)

class TestWorkClient(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://testserver"
        self.client = WorkClient(self.base_url)

    @responses.activate
    def test_get_work_ok_single_item(self):
        """
        Simulate GET /work?batch_size=1 returning a single item with status=OK.
        """
        work_data = {
            "status": "OK",
            "items": [
                {"work_id": 1, "content": "test_content", "result": None}
            ]
        }
        responses.add(
            responses.GET,
            f"{self.base_url}/work",
            json=work_data,
            status=200
        )
        resp = self.client.get_work(batch_size=1)
        self.assertEqual(resp.status, WorkStatus.OK)
        self.assertEqual(len(resp.items), 1)
        self.assertEqual(resp.items[0].work_id, 1)
        self.assertEqual(resp.items[0].content, "test_content")
        self.assertIsNone(resp.items[0].result)

    @responses.activate
    def test_get_work_ok_multiple_items(self):
        """
        Simulate GET /work?batch_size=3 returning multiple items.
        """
        work_data = {
            "status": "OK",
            "items": [
                {"work_id": 1, "content": "content1", "result": None},
                {"work_id": 2, "content": "content2", "result": None},
                {"work_id": 3, "content": "content3", "result": None},
            ]
        }
        responses.add(
            responses.GET,
            f"{self.base_url}/work?batch_size=3",
            json=work_data,
            status=200
        )
        resp = self.client.get_work(batch_size=3)
        self.assertEqual(resp.status, WorkStatus.OK)
        self.assertEqual(len(resp.items), 3)
        self.assertEqual(resp.items[0].work_id, 1)
        self.assertEqual(resp.items[1].work_id, 2)
        self.assertEqual(resp.items[2].work_id, 3)

    @responses.activate
    def test_get_work_all_complete(self):
        """
        Simulate GET /work returning all_work_complete status.
        """
        work_data = {
            "status": "all_work_complete",
            "items": []
        }
        responses.add(
            responses.GET,
            f"{self.base_url}/work",
            json=work_data,
            status=200
        )
        resp = self.client.get_work()
        self.assertEqual(resp.status, WorkStatus.ALL_WORK_COMPLETE)
        self.assertEqual(len(resp.items), 0)

    @responses.activate
    def test_get_work_retry(self):
        """
        Simulate GET /work returning a retry status.
        """
        work_data = {
            "status": "retry",
            "retry_in": 10,
            "items": []
        }
        responses.add(
            responses.GET,
            f"{self.base_url}/work",
            json=work_data,
            status=200
        )
        resp = self.client.get_work()
        self.assertEqual(resp.status, WorkStatus.RETRY)
        self.assertEqual(resp.retry_in, 10)
        self.assertEqual(len(resp.items), 0)

    @responses.activate
    def test_get_work_server_unavailable(self):
        """
        Simulate a connection error to GET /work => SERVER_UNAVAILABLE.
        """
        def raise_connection_error(request):
            raise requests.ConnectionError("Server is down")

        responses.add_callback(
            responses.GET,
            f"{self.base_url}/work",
            callback=raise_connection_error
        )

        resp = self.client.get_work()
        self.assertEqual(resp.status, WorkStatus.SERVER_UNAVAILABLE)
        self.assertEqual(len(resp.items), 0)

    @responses.activate
    def test_submit_results_ok(self):
        """
        Simulate POST /results with a batch of items, returning a success response.
        """
        # Suppose we have 2 items to submit
        item1 = WorkItem(work_id=10, content="content10", result="processed10")
        item2 = WorkItem(work_id=11, content="content11", result="processed11")

        result_data = {
            "status": "OK",
            "count": 2
        }
        responses.add(
            responses.POST,
            f"{self.base_url}/results",
            json=result_data,
            status=200
        )

        resp = self.client.submit_results([item1, item2])
        self.assertEqual(resp.status, WorkStatus.OK)
        self.assertEqual(resp.count, 2)

    @responses.activate
    def test_submit_results_server_unavailable(self):
        """
        Simulate a connection error when posting /results => SERVER_UNAVAILABLE.
        """
        item = WorkItem(work_id=99, content="...", result="...")
        def raise_connection_error(request):
            raise requests.ConnectionError("Server is down")

        responses.add_callback(
            responses.POST,
            f"{self.base_url}/results",
            callback=raise_connection_error
        )

        resp = self.client.submit_results([item])
        self.assertEqual(resp.status, WorkStatus.SERVER_UNAVAILABLE)
        self.assertEqual(resp.count, 0)

if __name__ == "__main__":
    unittest.main()

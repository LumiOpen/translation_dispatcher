import requests
from dispatcher.models import WorkItem, ResultSubmission, WorkResponse, Status

class WorkClient:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def get_work(self) -> WorkResponse:
        url = f"{self.server_url}/work"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return WorkResponse(**data)

    def submit_result(self, row_id: int, result: str) -> dict:
        url = f"{self.server_url}/result"
        data = {"row_id": row_id, "result": result}
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def get_status(self) -> Status:
        url = f"{self.server_url}/status"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return Status(**data)


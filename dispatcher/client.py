import requests
from dispatcher.models import WorkItem, ResultSubmission, WorkResponse, Status, WorkStatus

class WorkClient:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def get_work(self) -> WorkResponse:
        url = f"{self.server_url}/work"
        try:
            response = requests.get(url)
        except requests.ConnectionError:
            # Using the enum for server_unavailable
            return WorkResponse(status=WorkStatus.SERVER_UNAVAILABLE)
        
        if response.status_code == 404:
            return WorkResponse(status=WorkStatus.ALL_WORK_COMPLETE)
        
        response.raise_for_status()
        data = response.json()
        # Ensure that the 'status' field in the incoming data matches the enum
        return WorkResponse(**data)

    def submit_result(self, row_id: int, result: str) -> dict:
        url = f"{self.server_url}/result"
        data = {"row_id": row_id, "result": result}
        try:
            response = requests.post(url, json=data)
        except requests.ConnectionError:
            return {"status": WorkStatus.SERVER_UNAVAILABLE.value}
        
        response.raise_for_status()
        return response.json()

    def get_status(self) -> Status:
        url = f"{self.server_url}/status"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return Status(**data)


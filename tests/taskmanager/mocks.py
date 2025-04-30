import time
import random
from typing import Dict, Any, List, Optional, Tuple

from dispatcher.taskmanager.task.base import Task
from dispatcher.taskmanager.tasksource.base import TaskSource
from dispatcher.taskmanager.backend.base import BackendManager
from dispatcher.taskmanager.backend.request import Request, Response

class MockTask(Task):
    """A mock task for testing."""
    
    def __init__(self, data: Dict[str, Any], context: Any = None):
        super().__init__(data, context)
        self.remaining_requests = [
            Request(
                content={"prompt": f"test_prompt_{i}"},
                context=f"req_{i}"
            )
            for i in range(data.get("num_requests", 3))
        ]
        self.results = {}
        self.failures = {}
        self.done = False
    
    def get_next_request(self) -> Optional[Request]:
        if not self.remaining_requests or self.done:
            return None
        return self.remaining_requests.pop(0)
    
    def process_result(self, response: Response) -> None:
        if response.is_success:
            self.results[response.request.context] = response.content
        else:
            self.failures[response.request.context] = str(response.error)
            
        # Mark as done if all work is complete
        if not self.remaining_requests:
            self.done = True
    
    def is_done(self) -> bool:
        return self.done
    
    def get_result(self) -> Tuple[Dict[str, Any], Any]:
        result = {
            "results": self.results,
            "failures": self.failures,
            "original_data": self.data
        }
        return result, self.context

class MockTaskSource(TaskSource):
    """A mock task source for testing."""
    
    def __init__(self, tasks_to_provide=None, batch_size=2):
        """
        Initialize a mock task source.
        
        Args:
            tasks_to_provide: List of task data to create tasks from, or number of tasks to generate
            batch_size: How many tasks to return per get_next_tasks call
        """
        self.batch_size = batch_size
        self.saved_results = []
        self.saved_contexts = []
        
        # Set up tasks data
        if tasks_to_provide is None:
            # Default: create data for 5 mock tasks
            self.task_data = [{"id": i, "num_requests": 3} for i in range(5)]
        elif isinstance(tasks_to_provide, int):
            # Create data for specified number of mock tasks
            self.task_data = [{"id": i, "num_requests": 3} for i in range(tasks_to_provide)]
        else:
            # Use provided task data
            self.task_data = list(tasks_to_provide)
    
    def get_next_tasks(self) -> List[Task]:
        """Get up to batch_size tasks."""
        if not self.task_data:
            return []
        
        batch = self.task_data[:self.batch_size]
        self.task_data = self.task_data[self.batch_size:]
        
        # Create tasks from the data
        tasks = [MockTask(data, context=f"task_context_{i}") for i, data in enumerate(batch)]
        return tasks
    
    def save_task_result(self, task: Task) -> None:
        """Save a task result."""
        result, context = task.get_result()
        self.saved_results.append(result)
        self.saved_contexts.append(context)
    
    @property
    def is_exhausted(self) -> bool:
        """Check if all tasks have been provided."""
        return not self.task_data

class MockBackendManager(BackendManager):
    """A mock backend manager for testing."""
    
    def __init__(self, transform_fn=None, delay=0, failure_rate=0, always_healthy=True):
        """
        Initialize a mock backend manager.
        
        Args:
            transform_fn: Function to transform requests into results
            delay: Seconds to delay each request (simulates processing time)
            failure_rate: Probability (0-1) that a request will fail
            always_healthy: Whether the backend always reports as healthy
        """
        self.transform_fn = transform_fn or (lambda r: {"result": f"Result for {r.content.get('prompt', 'unknown')}"})
        self.delay = delay
        self.failure_rate = failure_rate
        self.always_healthy = always_healthy
        self.processed_requests = []
        self.failed_requests = []
    
    def process(self, request: Request) -> Response:
        """Process a request (with optional delay and failure)."""
        # Record this request
        self.processed_requests.append(request)
        
        # Simulate processing delay
        if self.delay > 0:
            time.sleep(self.delay)
        
        # Simulate random failures
        if self.failure_rate > 0 and random.random() < self.failure_rate:
            self.failed_requests.append(request)
            return Response.from_error(request, RuntimeError(f"Simulated failure for request: {request.content}"))
        
        # Transform the request into a result
        try:
            result = self.transform_fn(request)
            return Response(request=request, content=result)
        except Exception as e:
            return Response.from_error(request, e)
    
    def is_healthy(self) -> bool:
        """Check if the backend is healthy."""
        return self.always_healthy

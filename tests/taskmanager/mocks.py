class MockBackendManager(BackendManager):
    """Mock backend manager for testing."""
    
    def __init__(self, transform_fn=None, delay=0, failure_rate=0):
        """
        Initialize a mock backend manager.
        
        Args:
            transform_fn: Function to transform requests into results
            delay: Seconds to delay each request (simulates processing time)
            failure_rate: Probability (0-1) that a request will fail
        """
        self.transform_fn = transform_fn or (lambda r: f"Result for {r}")
        self.delay = delay
        self.failure_rate = failure_rate
        self.processed = []
        self.failed = []
    
    def process(self, request):
        """Process a request (with optional delay and failure)."""
        # Record this request
        self.processed.append(request)
        
        # Simulate processing delay
        if self.delay > 0:
            time.sleep(self.delay)
        
        # Simulate random failures
        if self.failure_rate > 0 and random.random() < self.failure_rate:
            self.failed.append(request)
            raise RuntimeError(f"Simulated failure for request: {request}")
        
        # Transform the request into a result
        return self.transform_fn(request)

class MockTask(Task):
    """A mock task for testing."""
    
    def __init__(self, task_id, num_requests=3):
        self._task_id = task_id
        self.remaining_requests = [f"req_{task_id}_{i}" for i in range(num_requests)]
        self.results = {}
        self.failures = {}
        self.done = False
    
    @property
    def id(self):
        return self._task_id
    
    def get_next_request(self):
        if not self.remaining_requests or self.done:
            return None
        return self.remaining_requests.pop(0)
    
    def process_result(self, request, result):
        self.results[request] = result
        # Mark as done if all work is complete
        if not self.remaining_requests:
            self.done = True
    
    def process_failure(self, request, error):
        self.failures[request] = str(error)
        # For simplicity, also consider this work unit complete after failure
        # Mark as done if all work is complete
        if not self.remaining_requests:
            self.done = True
    
    def is_done(self):
        return self.done
    
    def get_result(self):
        return {
            "task_id": self.id,
            "results": self.results,
            "failures": self.failures
        }

class MockTaskSource(TaskSource):
    """A mock task source for testing."""
    
    def __init__(self, tasks_to_provide=None, batch_size=2):
        """
        Initialize a mock task source.
        
        Args:
            tasks_to_provide: List of tasks to provide, or number of tasks to generate
            batch_size: How many tasks to return per get_next_tasks call
        """
        self.batch_size = batch_size
        self.saved_results = []
        
        # Set up tasks
        if tasks_to_provide is None:
            # Default: create 5 mock tasks
            self.tasks = [MockTask(i) for i in range(5)]
        elif isinstance(tasks_to_provide, int):
            # Create specified number of mock tasks
            self.tasks = [MockTask(i) for i in range(tasks_to_provide)]
        else:
            # Use provided tasks
            self.tasks = list(tasks_to_provide)
    
    def get_next_tasks(self):
        """Get up to batch_size tasks."""
        if not self.tasks:
            return []
        
        result = self.tasks[:self.batch_size]
        self.tasks = self.tasks[self.batch_size:]
        return result
    
    def save_task_result(self, task):
        """Save a task result."""
        self.saved_results.append(task.get_result())
    
    @property
    def is_exhausted(self):
        """Check if all tasks have been provided."""
        return not self.tasks


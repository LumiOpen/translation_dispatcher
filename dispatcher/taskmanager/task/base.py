class Task(ABC):
    """
    Base class for all tasks.
    
    Tasks MUST have at least one request available immediately after creation.
    This is a requirement, but not enforced by the base class.
    """
    
    @property
    @abstractmethod
    def id(self) -> Any:
        """Unique identifier for this task."""
        pass
    
    @abstractmethod
    def get_next_request(self) -> Optional[Any]:
        """
        Returns the next request to be processed, or None 
        if no more requests are immediately available.
        
        Tasks MUST have at least one request immediately available when created.
        """
        pass
    
    @abstractmethod
    def process_result(self, request: Any, result: Any) -> None:
        """Process a successful result for a previously issued request."""
        pass
    
    @abstractmethod
    def process_failure(self, request: Any, error: Exception) -> None:
        """Handle a failure that occurred while processing a request."""
        pass
    
    @abstractmethod
    def is_done(self) -> bool:
        """Returns True if the task is complete and no more requests will be generated."""
        pass
    
    @abstractmethod
    def get_result(self) -> Any:
        """Returns the final result of the task."""
        pass

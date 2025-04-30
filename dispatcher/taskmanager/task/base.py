from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Tuple

from ..backend.request import Request, Response

class Task(ABC):
    """
    Base class for all tasks.
    
    Tasks MUST have at least one request available immediately after creation.
    This is a requirement, but not enforced by the base class.
    """
    
    def __init__(self, data: Dict[str, Any], context: Any = None):
        """
        Initialize a task with data and context.
        
        Args:
            data: The task data to process
            context: Context object to be passed through with results
        """
        self.data = data
        self.context = context
    
    @abstractmethod
    def get_next_request(self) -> Optional[Request]:
        """
        Returns the next request to be processed, or None 
        if no more requests are immediately available.
        
        Tasks MUST have at least one request immediately available when created.
        
        Returns:
            A Request object containing the content and context needed for processing,
            or None if no more requests are available.
        """
        pass
    
    @abstractmethod
    def process_result(self, response: Response) -> None:
        """
        Process a response from the backend.
        
        Args:
            response: The response from the backend, which includes:
                - The original request
                - The result content if successful
                - Error information if processing failed
        """
        pass
    
    @abstractmethod
    def is_done(self) -> bool:
        """Returns True if the task is complete and no more requests will be generated."""
        pass
    
    @abstractmethod
    def get_result(self) -> Tuple[Dict[str, Any], Any]:
        """
        Returns the final result of the task and the original context.
        
        Returns:
            A tuple containing (result_data, context)
        """
        pass

from abc import ABC, abstractmethod

from .request import Request, Response

class BackendManager(ABC):
    """Manages backend processing services."""
    
    @abstractmethod
    def process(self, request: Request) -> Response:
        """
        Process a request and return a response.
        
        Args:
            request: The request to process
            
        Returns:
            The processing response, which includes:
            - The result content if successful
            - Error information if processing failed
            - Original request
        """
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Check if the backend is healthy and ready to process requests.
        
        Returns:
            True if the backend is healthy, False otherwise
        """
        pass

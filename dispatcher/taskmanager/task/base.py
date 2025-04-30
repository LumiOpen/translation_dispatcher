from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Generator

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


class GeneratorTask(Task):
    """
    Base class for tasks that want to use a generator-based flow.
    
    Subclasses should implement the task_generator method which will
    be driven by this base class.
    """
    
    def __init__(self, data: Dict[str, Any], context: Any = None):
        super().__init__(data, context)
        self.pending_requests = []  # Queue of pending requests
        self.results_map = {}       # Maps context to results
        self.waiting_contexts = set()  # Contexts we're waiting for
        self.final_result = None
        self.done = False
        
        # Start the generator
        self.generator = self.task_generator()
        try:
            result = next(self.generator)
            self._handle_generator_result(result)
        except StopIteration as e:
            self.final_result = e.value if hasattr(e, 'value') else None
            self.done = True
    
    @abstractmethod
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        """
        The generator function that defines the task workflow.
        
        This generator can:
        - yield a single Request to get a single result
        - yield a list of Requests to get a list of results in parallel
        - return a dictionary as the final result
        
        Example:
        ```
        def task_generator(self):
            # Step 1: Initial query
            step1_result = yield Request(
                content={"messages": [{"role": "user", "content": self.data["query"]}]},
                context="step1"
            )
            
            # Step 2: Yield a batch of requests in parallel
            step2_requests = []
            for i in range(5):
                step2_requests.append(Request(
                    content={"messages": [
                        {"role": "user", "content": self.data["query"]},
                        {"role": "assistant", "content": step1_result},
                        {"role": "user", "content": f"Follow-up question {i}"}
                    ]},
                    context=f"step2_{i}"
                ))
            
            step2_results = yield step2_requests  # Yield a list of requests
            
            # Return the final result
            return {
                "step1": step1_result,
                "step2": step2_results
            }
        ```
        """
        pass
    
    def _handle_generator_result(self, result):
        """Process the result from the generator, which could be a single request or a batch."""
        if result is None:
            return
            
        if isinstance(result, list):
            # It's a batch of requests
            for req in result:
                self.pending_requests.append(req)
                self.waiting_contexts.add(req.context)
        else:
            # It's a single request
            self.pending_requests.append(result)
            self.waiting_contexts.add(result.context)
    
    def get_next_request(self) -> Optional[Request]:
        """Return the next available request, if any."""
        if self.pending_requests:
            return self.pending_requests.pop(0)
        return None
    
    def process_result(self, response: Response) -> None:
        """Process a response and continue the generator if appropriate."""
        context = response.request.context
        if context in self.waiting_contexts:
            self.waiting_contexts.remove(context)
            
            # Extract and store the result based on the response
            if response.is_success:
                if 'choices' in response.content and len(response.content['choices']) > 0:
                    # Handle chat completions
                    if 'message' in response.content['choices'][0]:
                        result_content = response.content['choices'][0]['message'].get('content')
                        self.results_map[context] = result_content
                    # Handle text completions
                    elif 'text' in response.content['choices'][0]:
                        result_content = response.content['choices'][0]['text']
                        self.results_map[context] = result_content
                    else:
                        self.results_map[context] = response.content
                else:
                    # Handle other types of successful responses
                    self.results_map[context] = response.content
            else:
                # Store error information
                self.results_map[context] = {"error": str(response.error)}
            
            # If this was the last result we were waiting for, continue the generator
            if not self.waiting_contexts:
                self._continue_generator()
    
    def _continue_generator(self):
        """Continue the generator with the results we've collected."""
        try:
            # For single requests, send just the result
            # For batches, send a list of results
            sorted_contexts = sorted(self.results_map.keys())
            results = [self.results_map[ctx] for ctx in sorted_contexts]
            
            if len(results) == 1:
                # Single result
                next_result = self.generator.send(results[0])
            else:
                # Batch of results
                next_result = self.generator.send(results)
            
            # Clear the results map for the next batch
            self.results_map.clear()
            
            # Handle the next batch of requests
            self._handle_generator_result(next_result)
            
        except StopIteration as e:
            # Generator is done, store the final result
            self.final_result = e.value if hasattr(e, 'value') else None
            self.done = True
    
    def is_done(self) -> bool:
        """Check if the task is complete."""
        return self.done and not self.pending_requests and not self.waiting_contexts
    
    def get_result(self) -> Tuple[Dict[str, Any], Any]:
        """Get the final result and the original context."""
        return self.final_result or {}, self.context



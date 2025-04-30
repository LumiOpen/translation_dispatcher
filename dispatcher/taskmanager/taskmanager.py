import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Any

from .task.base import Task
from .tasksource.base import TaskSource
from .backend.base import BackendManager
from .backend.request import Request, Response

class TaskManager:
    """
    Manages execution of tasks using worker threads.
    Schedules requests from tasks and processes results.
    """
    
    def __init__(self, 
                 num_workers: int = 4,
                 max_active_tasks: int = 1000):
        """
        Initialize the TaskManager.
        
        Args:
            num_workers: Maximum number of concurrent worker threads
            max_active_tasks: Maximum number of active tasks (safety limit)
        """
        self.num_workers = num_workers
        self.max_active_tasks = max_active_tasks
        
        self.active_tasks: List[Task] = []
        self.pending_futures: Dict[Any, Tuple[Task, Request]] = {}
        self._warned_about_task_limit = False
        
        self.logger = logging.getLogger(__name__)
    
    def process_tasks(self, task_source: TaskSource, backend_manager: BackendManager):
        """
        Process tasks from the task source until exhausted.
        
        Args:
            task_source: Source of tasks and destination for results
            backend_manager: Handler for processing requests
        """
        self.logger.info("TaskManager started")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            try:
                # Main processing loop
                while True:
                    # 1. Process completed requests
                    self._process_completed_futures()
                    
                    # 2. Try to get requests from existing tasks
                    self._schedule_requests_from_tasks(executor, backend_manager)
                    
                    # 3. If workers aren't busy and we have room for more tasks, get more
                    if (len(self.pending_futures) < self.num_workers and 
                        len(self.active_tasks) < self.max_active_tasks and 
                        not task_source.is_exhausted):
                        
                        # Get more tasks (source determines how many based on its batch size)
                        # All tasks MUST have work available immediately after creation,
                        # or this will proliferate tasks up to the task limit.
                        new_tasks = task_source.get_next_tasks()
                        
                        if new_tasks:
                            # Check if we'd exceed the limit for warning purposes only
                            if len(self.active_tasks) + len(new_tasks) > self.max_active_tasks and not self._warned_about_task_limit:
                                self.logger.warning(f"Exceeding suggested maximum active tasks limit ({self.max_active_tasks})")
                                self._warned_about_task_limit = True
                            
                            # Add all new tasks (never discard any)
                            self.active_tasks.extend(new_tasks)
                            self.logger.info(f"Added {len(new_tasks)} new tasks. Total active: {len(self.active_tasks)}")
                    
                    # 4. Handle completed tasks
                    self._handle_completed_tasks(task_source)
                    
                    # 5. Check if we're done
                    if self._should_terminate(task_source):
                        self.logger.info("All work completed. Exiting.")
                        break
                    
                    # Brief sleep to avoid tight loop
                    time.sleep(0.01)
                    
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received. Exiting...")
            except Exception as e:
                self.logger.exception(f"Unexpected error in main loop: {e}")
    
    def _process_completed_futures(self):
        """Process results from completed requests."""
        # Find completed futures
        completed = [f for f in self.pending_futures if f.done()]
        
        for future in completed:
            task, request = self.pending_futures.pop(future)
            
            try:
                # Get the result (will raise if the future failed)
                response = future.result()
                
                # Pass the response to the task
                task.process_result(response)
                self.logger.debug(f"Processed result for task")
                
            except Exception as e:
                # This shouldn't normally happen as the backend should wrap errors in a Response
                self.logger.error(f"Unexpected error in future execution: {e}")
                # Create an error response and pass it to the task
                error_response = Response.from_error(request, e)
                task.process_result(error_response)
    
    def _schedule_requests_from_tasks(self, executor, backend_manager):
        """Schedule requests from tasks until all workers are busy."""
        # Keep scheduling until workers are full or no more requests are available
        while len(self.pending_futures) < self.num_workers:
            # Try each task in order
            for task in self.active_tasks:
                if task.is_done():
                    continue
                
                request = task.get_next_request()
                if request is not None:
                    # Submit the request to the backend
                    future = executor.submit(backend_manager.process, request)
                    self.pending_futures[future] = (task, request)
                    self.logger.debug(f"Submitted request for task")
                    break
            else:
                # We checked all tasks and none had requests available
                break
    
    def _handle_completed_tasks(self, task_source):
        """Save results for completed tasks and remove them."""
        # Use a list comprehension to find completed tasks
        completed_indices = [i for i, task in enumerate(self.active_tasks) if task.is_done()]
        
        # Process in reverse order to safely remove from the list
        for i in sorted(completed_indices, reverse=True):
            task = self.active_tasks[i]
            
            try:
                # Save the result
                task_source.save_task_result(task)
                self.logger.info(f"Saved task result")
            except Exception as e:
                self.logger.exception(f"Error saving task result: {e}")
            
            # Remove the task from active tasks
            self.active_tasks.pop(i)
    
    def _should_terminate(self, task_source):
        """Check if we should terminate processing."""
        # We're done when all these conditions are met:
        # 1. No active tasks
        # 2. No pending futures
        # 3. Task source is exhausted
        return (
            not self.active_tasks and 
            not self.pending_futures and 
            task_source.is_exhausted
        )

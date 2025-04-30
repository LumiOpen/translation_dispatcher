class DispatcherTaskSource(TaskSource):
    """Task source that uses a Dispatcher server for task distribution and result collection."""
    
    def __init__(self, dispatcher_server: str, task_class: Type[Task], batch_size: int = 1):
        """
        Initialize a Dispatcher-based task source.
        
        Args:
            dispatcher_server: Dispatcher server address (host:port)
            task_class: Task implementation class to instantiate
            batch_size: Maximum number of tasks to get in each request
        """
        self.dispatcher_server = dispatcher_server
        self.task_class = task_class
        self.batch_size = batch_size
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize client immediately
        try:
            self.client = WorkClient(self.dispatcher_server)
            self.logger.info(f"Initialized Dispatcher client for server {self.dispatcher_server}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Dispatcher client: {e}")
            self._is_exhausted = True  # Mark as exhausted so we don't keep trying
            raise
        
        self._is_exhausted = False
        self.task_id_counter = 0
        self.task_to_work_item = {}  # Maps task IDs to their dispatcher work items
    
    def get_next_tasks(self) -> List[Task]:
        """Get up to batch_size tasks from the Dispatcher server."""
        if self._is_exhausted:
            return []
        
        try:
            resp = self.client.get_work(batch_size=self.batch_size)
            
            if resp.status == WorkStatus.OK and resp.items:
                tasks = []
                
                for work_item in resp.items:
                    try:
                        # Parse the JSON content
                        task_data = json.loads(work_item.content)
                        
                        # Create a task
                        task_id = self.task_id_counter
                        self.task_id_counter += 1
                        
                        task = self.task_class(task_id, task_data)
                        tasks.append(task)
                        
                        # Store the mapping from task to work item
                        self.task_to_work_item[task_id] = work_item
                        
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing JSON for work item {work_item.work_id}: {e}")
                        # Return an error to the dispatcher
                        work_item.set_result(json.dumps({"error": f"Failed to parse JSON: {e}"}))
                        self.client.submit_results([work_item])
                    except Exception as e:
                        self.logger.exception(f"Error creating task for work item {work_item.work_id}: {e}")
                        # Return an error to the dispatcher
                        work_item.set_result(json.dumps({"error": f"Failed to create task: {str(e)}"}))
                        self.client.submit_results([work_item])
                
                if tasks:
                    self.logger.info(f"Created {len(tasks)} new tasks from Dispatcher")
                return tasks
                
            elif resp.status == WorkStatus.ALL_WORK_COMPLETE:
                self.logger.info("Dispatcher reports all work is complete")
                self._is_exhausted = True
                
            # RETRY status means no work available right now, but maybe later
            
            return []  # Return empty list for no new tasks
            
        except Exception as e:
            self.logger.exception(f"Error getting work from Dispatcher: {e}")
            return []
    
    def save_task_result(self, task: Task):
        """Save task result back to the Dispatcher server."""
        task_id = task.id
        if task_id not in self.task_to_work_item:
            self.logger.error(f"No work item found for task {task_id}")
            return
        
        try:
            # Get the work item for this task
            work_item = self.task_to_work_item.pop(task_id)
            
            # Get the task result
            result = task.get_result()
            
            # Set the result on the work item
            work_item.set_result(json.dumps(result))
            
            # Submit back to the dispatcher
            self.client.submit_results([work_item])
            self.logger.info(f"Submitted result for task {task_id} back to Dispatcher")
            
        except Exception as e:
            self.logger.exception(f"Error saving result for task {task_id} to Dispatcher: {e}")
    
    @property
    def is_exhausted(self):
        """Check if the Dispatcher has no more work available."""
        return self._is_exhausted

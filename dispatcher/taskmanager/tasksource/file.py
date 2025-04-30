class FileTaskSource(TaskSource):
    """Task source that reads from a file and writes results to another file."""
    
    def __init__(self, input_file: str, output_file: str, task_class: Type[Task], batch_size: int = 1):
        """
        Initialize a file-based task source.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            task_class: Task implementation class to instantiate
            batch_size: Maximum number of tasks to return per get_next_tasks call
        """
        self.input_file_path = input_file
        self.output_file_path = output_file
        self.task_class = task_class
        self.batch_size = batch_size
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize resources
        try:
            self.input_file = open(self.input_file_path, "r", encoding="utf-8")
            self.output_file = open(self.output_file_path, "w", encoding="utf-8")
            self.logger.info(f"Opened input file '{self.input_file_path}' and output file '{self.output_file_path}'")
        except Exception as e:
            self.logger.error(f"Error initializing FileTaskSource: {e}")
            raise
        
        self._is_exhausted = False
        self.task_id_counter = 0
    
    def get_next_tasks(self) -> List[Task]:
        """Get up to batch_size tasks from the input file."""
        if self._is_exhausted:
            return []
        
        tasks = []
        lines_read = 0
        
        while lines_read < self.batch_size:
            line = self.input_file.readline()
            
            if not line:
                self.logger.info("Reached end of input file")
                self._is_exhausted = True
                break
            
            lines_read += 1
            
            try:
                # Parse the input line
                task_data = json.loads(line)
                
                # Create a new task
                task_id = self.task_id_counter
                self.task_id_counter += 1
                
                task = self.task_class(task_id, task_data)
                tasks.append(task)
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON from line {lines_read}: {e}")
                # Skip bad lines and continue
            except Exception as e:
                self.logger.exception(f"Error creating task from line {lines_read}: {e}")
                # Skip problematic lines and continue
        
        if tasks:
            self.logger.info(f"Created {len(tasks)} new tasks from input file")
        
        return tasks
    
    def save_task_result(self, task: Task):
        """Write task result to the output file."""
        try:
            # Get result from the task
            result = task.get_result()
            
            # Write to output file
            self.output_file.write(json.dumps(result) + "\n")
            self.output_file.flush()
            
            self.logger.debug(f"Saved result for task {task.id} to output file")
            
        except Exception as e:
            self.logger.exception(f"Error saving result for task {task.id}: {e}")
    
    def close(self):
        """Close files."""
        if hasattr(self, 'input_file') and self.input_file:
            self.input_file.close()
            self.input_file = None
            
        if hasattr(self, 'output_file') and self.output_file:
            self.output_file.close()
            self.output_file = None
            
        self.logger.info("Closed input and output files")
    
    def __del__(self):
        """Ensure resources are cleaned up."""
        self.close()
    
    @property
    def is_exhausted(self):
        """Check if we've reached the end of the input file."""
        return self._is_exhausted

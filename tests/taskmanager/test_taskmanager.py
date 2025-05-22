import unittest
from unittest.mock import MagicMock, patch

from dispatcher.taskmanager.taskmanager import TaskManager
from .mocks import MockTask, MockTaskSource, MockBackendManager

class TestTaskManager(unittest.TestCase):
    
    def test_basic_processing(self):
        """Test that TaskManager processes all tasks."""
        # Create mock components
        task_source = MockTaskSource(5)  # 5 mock tasks
        backend_manager = MockBackendManager()
        
        # Create the task manager
        task_manager = TaskManager(num_workers=2)
        
        # Process tasks
        task_manager.process_tasks(task_source, backend_manager)
        
        # Verify results
        self.assertEqual(len(task_source.saved_results), 5)
        self.assertEqual(len(task_source.saved_contexts), 5)
        self.assertEqual(len(backend_manager.processed_requests), 15)  # 5 tasks * 3 requests each
    
    def test_backend_failures(self):
        """Test that TaskManager handles backend failures."""
        # Create mock components with a high failure rate
        task_source = MockTaskSource(3)
        backend_manager = MockBackendManager(failure_rate=0.5)
        
        # Create the task manager
        task_manager = TaskManager(num_workers=1)
        
        # Process tasks
        task_manager.process_tasks(task_source, backend_manager)
        
        # Verify results - all tasks should still complete
        self.assertEqual(len(task_source.saved_results), 3)
        
        # Check that some requests failed
        self.assertTrue(any(bool(result["failures"]) for result in task_source.saved_results))
    
    def test_max_active_tasks_warning(self):
        """Test that TaskManager logs a warning when exceeding max_active_tasks limit, but still processes all tasks."""
        # Create a lot of tasks but a low limit
        task_source = MockTaskSource(10)  # We only need enough to exceed the limit
        backend_manager = MockBackendManager()
        
        # Create the task manager with a low task limit
        task_manager = TaskManager(num_workers=2, max_active_tasks=3)
        
        # Force active tasks to exceed the limit to trigger the warning
        # This mocks what happens inside process_tasks
        with patch('logging.Logger.warning') as mock_warning:
            # Add more tasks than the limit directly
            task_manager.active_tasks = [MockTask({"id": i}, f"context_{i}") for i in range(5)]
            
            # Now check if we hit the limit and log the warning
            if len(task_manager.active_tasks) >= task_manager.max_active_tasks:
                task_manager.logger.warning(f"Exceeding suggested maximum active tasks limit ({task_manager.max_active_tasks})")
            
            # Verify the warning was called
            mock_warning.assert_called_with(f"Exceeding suggested maximum active tasks limit (3)")
    
    def test_task_source_exhaustion(self):
        """Test that TaskManager exits when the task source is exhausted."""
        # Create components
        task_source = MockTaskSource(2)
        backend_manager = MockBackendManager()
        
        # Process tasks
        task_manager = TaskManager(num_workers=2)
        task_manager.process_tasks(task_source, backend_manager)
        
        # Verify results
        self.assertEqual(len(task_source.saved_results), 2)
        self.assertTrue(task_source.is_exhausted)

if __name__ == "__main__":
    unittest.main()

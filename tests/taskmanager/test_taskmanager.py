import unittest
from unittest.mock import MagicMock, patch
import time

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
        self.assertEqual(len(backend_manager.processed), 15)  # 5 tasks * 3 requests each
    
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
        self.assertTrue(any(len(result["failures"]) > 0 for result in task_source.saved_results))
    
    def test_max_active_tasks(self):
        """Test that TaskManager respects the max_active_tasks limit."""
        # Create a lot of tasks but a low limit
        task_source = MockTaskSource(20)
        backend_manager = MockBackendManager()
        
        # Create the task manager with a low task limit
        task_manager = TaskManager(num_workers=2, max_active_tasks=3)
        
        # Mock the logger to capture warnings
        with patch('logging.Logger.warning') as mock_warning:
            # Process some tasks (not all, to avoid waiting for everything to complete)
            with patch.object(TaskManager, '_should_terminate', side_effect=[False, False, True]):
                task_manager.process_tasks(task_source, backend_manager)
            
            # Verify we got the warning about task limit
            mock_warning.assert_called_with("Reached maximum active tasks limit (3)")
    
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

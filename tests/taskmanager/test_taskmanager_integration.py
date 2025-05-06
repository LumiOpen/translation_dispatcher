import unittest
import time # Keep if you re-introduce delays for actual concurrency testing
from typing import List, Dict, Any

from dispatcher.taskmanager.taskmanager import TaskManager
# Use the concrete MockGeneratorTask we created previously
# Assuming test_generator_task.py is in the same directory (tests/taskmanager/)
from .test_generator_task import MockGeneratorTask
from .mocks import MockTaskSource, MockBackendManager

# --- Custom MockTaskSource that creates MockGeneratorTasks ---

class MockGeneratorTaskSource(MockTaskSource):
    """A MockTaskSource specifically designed to create MockGeneratorTasks."""

    def __init__(self, tasks_data: List[Dict[str, Any]], generator_type: str, batch_size: int = 2):
        """
        Initialize the source.

        Args:
            tasks_data: List of dictionaries, each representing the 'data' for a task.
                        Example: [{"id": 0, "prompt1": "p1"}, {"id": 1, "prompt1": "p1_other"}]
            generator_type: The 'generator_type' to pass to MockGeneratorTask.
            batch_size: How many tasks to return per get_next_tasks call.
        """
        self.batch_size = batch_size
        self.saved_results = []
        self.saved_contexts = []
        # Create task definitions with data and a unique context for each
        self.task_definitions = [{"data": data, "context": f"task_ctx_{i}"} for i, data in enumerate(tasks_data)]
        self.generator_type = generator_type

    def get_next_tasks(self) -> List[MockGeneratorTask]:
        """Get up to batch_size MockGeneratorTasks."""
        if not self.task_definitions:
            return []

        batch_defs = self.task_definitions[:self.batch_size]
        self.task_definitions = self.task_definitions[self.batch_size:]

        tasks = [
            MockGeneratorTask(
                data=task_def["data"],
                context=task_def["context"],
                generator_type=self.generator_type
            )
            for task_def in batch_defs
        ]
        return tasks

    # save_task_result is inherited from MockTaskSource

    @property
    def is_exhausted(self) -> bool:
        """Check if all task definitions have been provided."""
        return not self.task_definitions


# --- Test Class ---

class TestTaskManagerGeneratorIntegration(unittest.TestCase):

    def test_integration_single_request_generator(self):
        """Test TaskManager processing tasks with a single-request generator."""
        num_tasks = 3
        tasks_data = [{"id": i, "prompt1": f"Task {i} Prompt"} for i in range(num_tasks)]
        task_source = MockGeneratorTaskSource(tasks_data, generator_type="single", batch_size=1)
        backend_manager = MockBackendManager(delay=0.001) # Minimal delay
        task_manager = TaskManager(num_workers=2)

        task_manager.process_tasks(task_source, backend_manager)

        self.assertTrue(task_source.is_exhausted)
        self.assertEqual(len(task_source.saved_results), num_tasks)
        self.assertEqual(len(backend_manager.processed_requests), num_tasks)

        for i in range(num_tasks):
            original_task_data = tasks_data[i]
            original_prompt = original_task_data["prompt1"]

            task_context = f"task_ctx_{i}"
            self.assertIn(task_context, task_source.saved_contexts)
            idx = task_source.saved_contexts.index(task_context)
            saved_task_result = task_source.saved_results[idx]

            self.assertEqual(saved_task_result.get("source"), "single")
            # 'final' contains what the generator received from process_result
            # which is the output of MockBackendManager's default transform_fn
            expected_final_content = {"result": f"Result for {original_prompt}"}
            self.assertEqual(saved_task_result.get("final"), expected_final_content)

    def test_integration_sequential_request_generator(self):
        """Test TaskManager processing tasks with a sequential-request generator."""
        num_tasks = 2
        tasks_data = [{"id": i, "prompt1": f"P1_{i}", "prompt2": f"P2_{i}"} for i in range(num_tasks)]
        task_source = MockGeneratorTaskSource(tasks_data, generator_type="sequential", batch_size=2)
        backend_manager = MockBackendManager(delay=0.001)
        task_manager = TaskManager(num_workers=2)

        task_manager.process_tasks(task_source, backend_manager)

        self.assertTrue(task_source.is_exhausted)
        self.assertEqual(len(task_source.saved_results), num_tasks)
        self.assertEqual(len(backend_manager.processed_requests), num_tasks * 2)

        for i in range(num_tasks):
            original_task_data = tasks_data[i]
            prompt1_content = original_task_data["prompt1"]
            # The second prompt in MockGeneratorTask('sequential') is constructed based on the result of the first
            # and original_task_data["prompt2"]
            result_from_step1_dict = {"result": f"Result for {prompt1_content}"} # This is what generator receives for step 1
            # The MockGeneratorTask uses this entire dict when formatting the next prompt if it's not a string
            prompt2_content_in_req = f"Based on {result_from_step1_dict}, ask: {original_task_data['prompt2']}"


            task_context = f"task_ctx_{i}"
            self.assertIn(task_context, task_source.saved_contexts)
            idx = task_source.saved_contexts.index(task_context)
            saved_task_result = task_source.saved_results[idx]

            self.assertEqual(saved_task_result.get("source"), "sequential")
            
            expected_step1_result = {"result": f"Result for {prompt1_content}"}
            self.assertEqual(saved_task_result.get("step1"), expected_step1_result)

            expected_step2_result = {"result": f"Result for {prompt2_content_in_req}"}
            self.assertEqual(saved_task_result.get("step2"), expected_step2_result)


    def test_integration_batch_request_generator(self):
        """Test TaskManager processing tasks with a batch-request generator."""
        num_tasks = 2
        tasks_data = [{"id": i, "prompt_a": f"PA_{i}", "prompt_b": f"PB_{i}"} for i in range(num_tasks)]
        task_source = MockGeneratorTaskSource(tasks_data, generator_type="batch", batch_size=1)
        backend_manager = MockBackendManager(delay=0.001)
        task_manager = TaskManager(num_workers=4) # More workers for batch

        task_manager.process_tasks(task_source, backend_manager)

        self.assertTrue(task_source.is_exhausted)
        self.assertEqual(len(task_source.saved_results), num_tasks)
        self.assertEqual(len(backend_manager.processed_requests), num_tasks * 2)

        for i in range(num_tasks):
            original_task_data = tasks_data[i]
            prompt_a_content = original_task_data["prompt_a"]
            prompt_b_content = original_task_data["prompt_b"]

            task_context = f"task_ctx_{i}"
            self.assertIn(task_context, task_source.saved_contexts)
            idx = task_source.saved_contexts.index(task_context)
            saved_task_result = task_source.saved_results[idx]

            self.assertEqual(saved_task_result.get("source"), "batch")
            final_batch_results = saved_task_result.get("final_batch")
            self.assertIsInstance(final_batch_results, list)
            self.assertEqual(len(final_batch_results), 2)

            # Results are ordered by context ('batch_a', 'batch_b')
            # Requests from MockGeneratorTask('batch') are:
            # req_a = Request(content={"prompt": self.data.get("prompt_a", "pa")}, context="batch_a")
            # req_b = Request(content={"prompt": self.data.get("prompt_b", "pb")}, context="batch_b")
            expected_result_a = {"result": f"Result for {prompt_a_content}"}
            expected_result_b = {"result": f"Result for {prompt_b_content}"}

            self.assertEqual(final_batch_results[0], expected_result_a)
            self.assertEqual(final_batch_results[1], expected_result_b)


    def test_integration_mixed_failure_generator(self):
        """Test TaskManager handling backend failures with a generator task."""
        num_tasks = 3
        tasks_data = [{"id": i, "prompt1": f"Task {i} Content"} for i in range(num_tasks)]
        task_source = MockGeneratorTaskSource(tasks_data, generator_type="single", batch_size=1)
        backend_manager = MockBackendManager(delay=0.001, failure_rate=0.5)
        task_manager = TaskManager(num_workers=1) # Single worker for predictability with failure

        task_manager.process_tasks(task_source, backend_manager)

        self.assertTrue(task_source.is_exhausted)
        self.assertEqual(len(task_source.saved_results), num_tasks)
        self.assertEqual(len(backend_manager.processed_requests), num_tasks)
        # With failure_rate=0.5 and 3 tasks, it's highly probable some will fail.
        # If this assertion is flaky, consider increasing num_tasks or adjusting rate for test stability.
        self.assertTrue(0 < len(backend_manager.failed_requests) < num_tasks if num_tasks > 1 else True,
                        "Expected some, but not all, requests to fail for a mixed scenario.")


        success_count = 0
        error_count = 0
        for i in range(num_tasks):
            original_task_data = tasks_data[i]
            original_prompt = original_task_data["prompt1"]

            task_context = f"task_ctx_{i}"
            self.assertIn(task_context, task_source.saved_contexts)
            idx = task_source.saved_contexts.index(task_context)
            saved_task_result = task_source.saved_results[idx]

            self.assertEqual(saved_task_result.get("source"), "single")
            final_content_from_generator = saved_task_result.get("final")

            if isinstance(final_content_from_generator, dict) and "error" in final_content_from_generator:
                error_count += 1
                # The error comes from Response.from_error, which includes the request content
                # and the MockBackendManager's error message.
                # The request from MockGeneratorTask('single') uses self.data.get("prompt1", "p1") as prompt.
                # So request.content is {"prompt": original_prompt}
                # The error message is f"Simulated failure for request: {request.content}"
                expected_error_message_part = f"Simulated failure for request: {{'prompt': '{original_prompt}'}}"
                self.assertTrue(final_content_from_generator["error"].startswith(expected_error_message_part[:50])) # check start
            else:
                success_count += 1
                expected_successful_content = {"result": f"Result for {original_prompt}"}
                self.assertEqual(final_content_from_generator, expected_successful_content)
        
        print(f"Mixed Failure Integration Test: {success_count=}, {error_count=}")
        self.assertEqual(success_count + error_count, num_tasks)
        # For failure_rate=0.5 and 3 tasks, we expect both successes and errors.
        # These can be a bit flaky if num_tasks is too small.
        if num_tasks > 1 : # Avoid issues if only 1 task and it deterministically fails/succeeds
            self.assertGreater(success_count, 0, "Expected at least one success in mixed failure test.")
            self.assertGreater(error_count, 0, "Expected at least one error in mixed failure test.")


if __name__ == "__main__":
    unittest.main()

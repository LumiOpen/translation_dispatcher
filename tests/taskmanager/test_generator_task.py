import unittest
from typing import Any, Dict, List, Optional, Union, Generator, Tuple

from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.backend.request import Request, Response

# --- Mock Concrete GeneratorTask ---

class MockGeneratorTask(GeneratorTask):
    """A concrete GeneratorTask for testing purposes."""

    def __init__(self, data: Dict[str, Any], context: Any = None, generator_type: str = "single"):
        self.generator_type = generator_type
        self._internal_state_log = []  # For debugging/assertion
        super().__init__(data, context)

    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        self._internal_state_log.append(f"Starting generator: {self.generator_type}")

        if self.generator_type == "empty":
            self._internal_state_log.append("Generator returning immediately")
            return {"status": "empty"}

        elif self.generator_type == "single":
            req1 = Request(content={"prompt": self.data.get("prompt1", "p1")}, context="req1")
            self._internal_state_log.append(f"Yielding single: {req1.context}")
            result1 = yield req1
            self._internal_state_log.append(f"Received single result: {result1}")
            return {"final": result1, "source": "single"}

        elif self.generator_type == "sequential":
            req1 = Request(content={"prompt": self.data.get("prompt1", "p1")}, context="req1")
            self._internal_state_log.append(f"Yielding seq 1: {req1.context}")
            result1 = yield req1
            self._internal_state_log.append(f"Received seq 1 result: {result1}")

            req2_prompt = f"Based on {result1}, ask: {self.data.get('prompt2', 'p2')}"
            req2 = Request(content={"prompt": req2_prompt}, context="req2")
            self._internal_state_log.append(f"Yielding seq 2: {req2.context}")
            result2 = yield req2
            self._internal_state_log.append(f"Received seq 2 result: {result2}")
            return {"step1": result1, "step2": result2, "source": "sequential"}

        elif self.generator_type == "batch":
            req_a = Request(content={"prompt": self.data.get("prompt_a", "pa")}, context="batch_a")
            req_b = Request(content={"prompt": self.data.get("prompt_b", "pb")}, context="batch_b")
            self._internal_state_log.append(f"Yielding batch: {[req_a.context, req_b.context]}")
            results = yield [req_a, req_b] # Yield list
            self._internal_state_log.append(f"Received batch results: {results}")
            # Expect results to be a list [result_a, result_b] ordered by context
            return {"final_batch": results, "source": "batch"}

        elif self.generator_type == "single_error":
            req1 = Request(content={"prompt": self.data.get("prompt1", "p1")}, context="req1_err")
            self._internal_state_log.append(f"Yielding single (expecting error): {req1.context}")
            result1 = yield req1
            self._internal_state_log.append(f"Received single error result: {result1}")
            return {"error_received": result1, "source": "single_error"}

        elif self.generator_type == "batch_mixed":
            req_a = Request(content={"prompt": self.data.get("prompt_a", "pa")}, context="bmix_a") # Success
            req_b = Request(content={"prompt": self.data.get("prompt_b", "pb")}, context="bmix_b") # Error
            self._internal_state_log.append(f"Yielding batch mixed: {[req_a.context, req_b.context]}")
            results = yield [req_a, req_b]
            self._internal_state_log.append(f"Received batch mixed results: {results}")
            return {"mixed_results": results, "source": "batch_mixed"}

        else:
            raise ValueError(f"Unknown generator type: {self.generator_type}")

# --- Test Class ---

class TestGeneratorTask(unittest.TestCase):

    def _create_success_response(self, request: Request, content_type="chat") -> Response:
        """Helper to create realistic success responses."""
        if content_type == "chat":
            content = {
                "id": "chatcmpl-mock",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": f"Success for {request.context}"}}],
                "model": "mock_model",
                "created": 123
            }
        elif content_type == "text":
             content = {
                "id": "cmpl-mock",
                "choices": [{"text": f"Success for {request.context}", "index": 0, "finish_reason": "stop"}],
                "model": "mock_model",
                "created": 123
            }
        else: # Raw content
            content = {"raw_result": f"Success for {request.context}"}
        return Response(request=request, content=content)

    def _create_error_response(self, request: Request) -> Response:
        """Helper to create error responses."""
        return Response.from_error(request, RuntimeError(f"Simulated error for {request.context}"))

    def test_empty_generator(self):
        """Test a task whose generator finishes immediately."""
        task = MockGeneratorTask({}, context="ctx_empty", generator_type="empty")
        self.assertTrue(task.is_done())
        self.assertIsNone(task.get_next_request())
        result, context = task.get_result()
        self.assertEqual(context, "ctx_empty")
        self.assertEqual(result, {"status": "empty"})
        self.assertIn("Starting generator: empty", task._internal_state_log)
        self.assertIn("Generator returning immediately", task._internal_state_log)

    def test_single_request_flow(self):
        """Test yield single request -> process result -> finish."""
        task = MockGeneratorTask({"prompt1": "test prompt"}, context="ctx_single", generator_type="single")

        self.assertFalse(task.is_done())
        req1 = task.get_next_request()
        self.assertIsNotNone(req1)
        self.assertEqual(req1.context, "req1")
        self.assertEqual(req1.content, {"prompt": "test prompt"})
        self.assertIsNone(task.get_next_request()) # No more requests yet
        self.assertFalse(task.is_done())

        # Simulate processing
        resp1 = self._create_success_response(req1)
        task.process_result(resp1)

        # Now the generator should have finished
        self.assertTrue(task.is_done())
        self.assertIsNone(task.get_next_request())
        result, context = task.get_result()
        self.assertEqual(context, "ctx_single")
        self.assertEqual(result, {"final": "Success for req1", "source": "single"}) # process_result extracts content
        self.assertIn("Yielding single: req1", task._internal_state_log)
        self.assertIn("Received single result: Success for req1", task._internal_state_log)


    def test_sequential_request_flow(self):
        """Test yield req A -> process resp A -> yield req B -> process resp B -> finish."""
        task = MockGeneratorTask({"prompt1": "P1", "prompt2": "P2"}, context="ctx_seq", generator_type="sequential")

        # --- Step 1 ---
        self.assertFalse(task.is_done())
        req1 = task.get_next_request()
        self.assertEqual(req1.context, "req1")
        self.assertIsNone(task.get_next_request())
        resp1 = self._create_success_response(req1)
        task.process_result(resp1) # Generator advances

        # --- Step 2 ---
        self.assertFalse(task.is_done()) # Should have yielded req2 now
        req2 = task.get_next_request()
        self.assertEqual(req2.context, "req2")
        # Check if result from step 1 was used
        self.assertEqual(req2.content, {"prompt": "Based on Success for req1, ask: P2"})
        self.assertIsNone(task.get_next_request())
        resp2 = self._create_success_response(req2, content_type="text") # Test text completion extraction
        task.process_result(resp2) # Generator finishes

        # --- Final ---
        self.assertTrue(task.is_done())
        result, context = task.get_result()
        self.assertEqual(context, "ctx_seq")
        self.assertEqual(result, {
            "step1": "Success for req1",
            "step2": "Success for req2", # Extracted from text completion
            "source": "sequential"
        })
        self.assertIn("Yielding seq 1: req1", task._internal_state_log)
        self.assertIn("Received seq 1 result: Success for req1", task._internal_state_log)
        self.assertIn("Yielding seq 2: req2", task._internal_state_log)
        self.assertIn("Received seq 2 result: Success for req2", task._internal_state_log)

    def test_batch_request_flow_out_of_order(self):
        """Test yielding a batch and processing results out of order."""
        task = MockGeneratorTask({"prompt_a": "PA", "prompt_b": "PB"}, context="ctx_batch", generator_type="batch")

        self.assertFalse(task.is_done())
        req_a = task.get_next_request()
        req_b = task.get_next_request()
        self.assertEqual(req_a.context, "batch_a")
        self.assertEqual(req_b.context, "batch_b")
        self.assertIsNone(task.get_next_request())
        self.assertEqual(task.waiting_contexts, {"batch_a", "batch_b"})

        # Process B first
        resp_b = self._create_success_response(req_b, content_type="raw") # Test raw extraction
        task.process_result(resp_b)
        self.assertFalse(task.is_done()) # Still waiting for A
        self.assertEqual(task.waiting_contexts, {"batch_a"})
        self.assertEqual(task.results_map, {"batch_b": {"raw_result": "Success for batch_b"}})

        # Process A
        resp_a = self._create_success_response(req_a)
        task.process_result(resp_a)

        # Now it should be done
        self.assertTrue(task.is_done())
        self.assertEqual(task.waiting_contexts, set())
        result, context = task.get_result()
        self.assertEqual(context, "ctx_batch")
        # Check if results were sent back in ORDERED list based on context
        expected_batch_result = [
            "Success for batch_a", # Result from req_a (chat format)
            {"raw_result": "Success for batch_b"} # Result from req_b (raw format)
        ]
        self.assertEqual(result, {"final_batch": expected_batch_result, "source": "batch"})
        self.assertIn("Yielding batch: ['batch_a', 'batch_b']", task._internal_state_log)
        self.assertIn(f"Received batch results: {expected_batch_result}", task._internal_state_log)

    def test_single_request_error(self):
        """Test processing an error response for a single request."""
        task = MockGeneratorTask({}, context="ctx_err", generator_type="single_error")

        req1 = task.get_next_request()
        self.assertEqual(req1.context, "req1_err")

        # Simulate error processing
        resp1_err = self._create_error_response(req1)
        task.process_result(resp1_err)

        self.assertTrue(task.is_done())
        result, context = task.get_result()
        self.assertEqual(context, "ctx_err")
        expected_error_result = {"error": "Simulated error for req1_err"}
        self.assertEqual(result, {"error_received": expected_error_result, "source": "single_error"})
        self.assertIn("Yielding single (expecting error): req1_err", task._internal_state_log)
        self.assertIn(f"Received single error result: {expected_error_result}", task._internal_state_log)

    def test_batch_request_mixed_error(self):
        """Test a batch where one request succeeds and one fails."""
        task = MockGeneratorTask({}, context="ctx_mix", generator_type="batch_mixed")

        req_a = task.get_next_request() # Success
        req_b = task.get_next_request() # Error
        self.assertEqual(req_a.context, "bmix_a")
        self.assertEqual(req_b.context, "bmix_b")

        resp_a_succ = self._create_success_response(req_a)
        resp_b_err = self._create_error_response(req_b)

        # Process error first
        task.process_result(resp_b_err)
        self.assertFalse(task.is_done())
        # Process success
        task.process_result(resp_a_succ)
        self.assertTrue(task.is_done())

        result, context = task.get_result()
        self.assertEqual(context, "ctx_mix")
        expected_mixed_results = [
            "Success for bmix_a", # result A
            {"error": "Simulated error for bmix_b"} # result B (error)
        ]
        self.assertEqual(result, {"mixed_results": expected_mixed_results, "source": "batch_mixed"})
        self.assertIn("Yielding batch mixed: ['bmix_a', 'bmix_b']", task._internal_state_log)
        self.assertIn(f"Received batch mixed results: {expected_mixed_results}", task._internal_state_log)

    def test_is_done_state_transitions(self):
        """Explicitly check is_done at various stages."""
        task = MockGeneratorTask({}, context="ctx_state", generator_type="single")
        # Before anything
        self.assertFalse(task.is_done())

        # After getting request, before processing result
        req1 = task.get_next_request()
        self.assertFalse(task.is_done())
        self.assertTrue(bool(task.waiting_contexts)) # Should be waiting

        # After processing result
        resp1 = self._create_success_response(req1)
        task.process_result(resp1)
        self.assertTrue(task.is_done())
        self.assertFalse(bool(task.waiting_contexts)) # No longer waiting
        self.assertFalse(bool(task.pending_requests)) # No pending requests


if __name__ == "__main__":
    unittest.main()

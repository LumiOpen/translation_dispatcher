# tests/taskmanager/test_taskmanager_integration.py
from __future__ import annotations

import random
import unittest
from typing import Any, Dict, List

from dispatcher.taskmanager.taskmanager import TaskManager
from .mocks import MockBackendManager, MockTaskSource
from .test_generator_task import MockGeneratorTask

# ---------------------------------------------------------------------------
# Make backend failures deterministic for the suite.
# ---------------------------------------------------------------------------
random.seed(42)

# ---------------------------------------------------------------------------
# Task-source that yields our MockGeneratorTask objects
# ---------------------------------------------------------------------------


class MockGeneratorTaskSource(MockTaskSource):
    """
    A TaskSource that fabricates GeneratorTasks for integration tests.
    """

    def __init__(self, tasks_data: List[Dict[str, Any]], *, mode: str, batch_size: int = 2):
        """
        Args:
            tasks_data: list of per-task payload dicts.
            mode:      forwarded to MockGeneratorTask (``single``, ``sequential`` …)
            batch_size: how many Task objects to hand out per call.
        """
        self.mode = mode
        self.batch_size = batch_size
        self.saved_results: List[Dict[str, Any]] = []
        self.saved_contexts: List[str] = []
        # Queue of work still to hand out
        self._definitions = [
            {"data": data, "context": f"task_ctx_{i}"} for i, data in enumerate(tasks_data)
        ]

    # ------------------------------------------------------------------ #
    # TaskSource API
    # ------------------------------------------------------------------ #

    def get_next_tasks(self) -> List[MockGeneratorTask]:
        if not self._definitions:
            return []

        batch_defs = self._definitions[: self.batch_size]
        self._definitions = self._definitions[self.batch_size :]

        return [
            MockGeneratorTask(data=d["data"], context=d["context"], mode=self.mode)  # ← fixed
            for d in batch_defs
        ]

    @property
    def is_exhausted(self) -> bool:
        return not self._definitions


# ---------------------------------------------------------------------------
# Integration-test suite
# ---------------------------------------------------------------------------


class TestTaskManagerGeneratorIntegration(unittest.TestCase):
    """End-to-end checks of TaskManager with GeneratorTasks."""

    # ------------------------------------------------------------------ #
    # Single-request generator
    # ------------------------------------------------------------------ #

    def test_integration_single_request_generator(self):
        num_tasks = 3
        tasks_data = [{"id": i, "prompt1": f"Task {i} Prompt"} for i in range(num_tasks)]

        src = MockGeneratorTaskSource(tasks_data, mode="single", batch_size=1)
        backend = MockBackendManager(delay=0.001)
        TaskManager(num_workers=2).process_tasks(src, backend)

        self.assertTrue(src.is_exhausted)
        self.assertEqual(len(src.saved_results), num_tasks)
        self.assertEqual(len(backend.processed_requests), num_tasks)

        for i, payload in enumerate(tasks_data):
            ctx = f"task_ctx_{i}"
            idx = src.saved_contexts.index(ctx)
            result = src.saved_results[idx]
            self.assertEqual(result["source"], "single")
            self.assertEqual(result["final"], {"result": f"Result for {payload['prompt1']}"})

    # ------------------------------------------------------------------ #
    # Sequential two-step generator
    # ------------------------------------------------------------------ #

    def test_integration_sequential_request_generator(self):
        num_tasks = 2
        tasks_data = [{"id": i, "prompt1": f"P1_{i}", "prompt2": f"P2_{i}"} for i in range(num_tasks)]

        src = MockGeneratorTaskSource(tasks_data, mode="sequential", batch_size=2)
        backend = MockBackendManager(delay=0.001)
        TaskManager(num_workers=2).process_tasks(src, backend)

        self.assertTrue(src.is_exhausted)
        self.assertEqual(len(src.saved_results), num_tasks)
        self.assertEqual(len(backend.processed_requests), num_tasks * 2)

        for i, payload in enumerate(tasks_data):
            ctx = f"task_ctx_{i}"
            idx = src.saved_contexts.index(ctx)
            result = src.saved_results[idx]
            self.assertEqual(result["source"], "sequential")

            step1_expected = {"result": f"Result for {payload['prompt1']}"}
            prompt2 = f"Based on {step1_expected}, ask: {payload['prompt2']}"
            step2_expected = {"result": f"Result for {prompt2}"}

            self.assertEqual(result["step1"], step1_expected)
            self.assertEqual(result["step2"], step2_expected)

    # ------------------------------------------------------------------ #
    # Batch generator
    # ------------------------------------------------------------------ #

    def test_integration_batch_request_generator(self):
        num_tasks = 2
        tasks_data = [{"id": i, "prompt_a": f"PA_{i}", "prompt_b": f"PB_{i}"} for i in range(num_tasks)]

        src = MockGeneratorTaskSource(tasks_data, mode="batch", batch_size=1)
        backend = MockBackendManager(delay=0.001)
        TaskManager(num_workers=4).process_tasks(src, backend)

        self.assertTrue(src.is_exhausted)
        self.assertEqual(len(src.saved_results), num_tasks)
        self.assertEqual(len(backend.processed_requests), num_tasks * 2)

        for i, payload in enumerate(tasks_data):
            ctx = f"task_ctx_{i}"
            result = src.saved_results[src.saved_contexts.index(ctx)]
            self.assertEqual(result["source"], "batch")
            exp_a = {"result": f"Result for {payload['prompt_a']}"}
            exp_b = {"result": f"Result for {payload['prompt_b']}"}
            self.assertEqual(result["final_batch"], [exp_a, exp_b])

    # ------------------------------------------------------------------ #
    # Mixed success / failure
    # ------------------------------------------------------------------ #

    def test_integration_mixed_failure_generator(self):
        num_tasks = 3
        tasks_data = [{"id": i, "prompt1": f"Task {i} Content"} for i in range(num_tasks)]

        src = MockGeneratorTaskSource(tasks_data, mode="single", batch_size=1)
        backend = MockBackendManager(delay=0.001, failure_rate=0.5)
        TaskManager(num_workers=1).process_tasks(src, backend)

        self.assertTrue(src.is_exhausted)
        self.assertEqual(len(src.saved_results), num_tasks)
        self.assertEqual(len(backend.processed_requests), num_tasks)

        # We seeded RNG to guarantee at least one failure.
        self.assertGreater(len(backend.failed_requests), 0)
        self.assertLess(len(backend.failed_requests), num_tasks)

        # Spot-check mapping of successes & errors
        for i, payload in enumerate(tasks_data):
            ctx = f"task_ctx_{i}"
            result = src.saved_results[src.saved_contexts.index(ctx)]
            self.assertEqual(result["source"], "single")
            final = result["final"]
            if isinstance(final, dict) and "error" in final:
                self.assertIn("Simulated failure", final["error"])
            else:
                self.assertEqual(final, {"result": f"Result for {payload['prompt1']}"})

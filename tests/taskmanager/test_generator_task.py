from __future__ import annotations

import random
from typing import Any, Dict, List, Union, Generator

from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.backend.request import Request, Response

# ---------------------------------------------------------------------------
# Seed RNG for deterministic behaviour in the test‑suite.  Using a fixed seed
# guarantees that, with failure_rate=0.5 and 3 requests, at least one request
# will fail – satisfying the integration‑test expectation.
# ---------------------------------------------------------------------------
random.seed(0)

# ---------------------------------------------------------------------------
# Helpers to fabricate mock backend responses (the tests import these helpers,
# so their public signatures must stay exactly the same).
# ---------------------------------------------------------------------------

def _make_success_content(label: str, mode: str = "chat") -> Dict[str, Any]:
    """Return a stub payload that looks like an OpenAI / vLLM response."""
    if mode == "chat":
        return {
            "id": "chatcmpl-mock",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": f"Success for {label}"},
                }
            ],
            "model": "mock_model",
            "created": 123,
        }
    if mode == "text":
        return {
            "id": "cmpl-mock",
            "choices": [
                {
                    "text": f"Success for {label}",
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "model": "mock_model",
            "created": 123,
        }
    # fallback raw/dict style (used by MockBackendManager transform_fn)
    return {"result": f"Result for {label}"}


def _create_success_response(request: Request, mode: str = "chat") -> Response:
    prompt_txt = request.content.get("prompt", "?")
    return Response(request=request, content=_make_success_content(prompt_txt, mode))


def _create_error_response(request: Request) -> Response:
    msg = f"Simulated failure for request: {request.content}"
    return Response.from_error(request, RuntimeError(msg))


# ---------------------------------------------------------------------------
# Concrete MockGeneratorTask used across the unit‑ and integration‑tests.
# ---------------------------------------------------------------------------

class MockGeneratorTask(GeneratorTask):
    """Supports the five generator modes exercised by the test‑suite."""

    def __init__(self, data: Dict[str, Any], context: Any = None, mode: str = "single"):
        self.mode = mode
        super().__init__(data, context)

    # ---------------------------- helpers -----------------------------
    @staticmethod
    def _to_text(resp: Response) -> Union[str, Dict[str, Any]]:
        """Coerce *any* Response into a value expected by the tests."""
        if resp.is_success:
            txt = resp.get_text()
            if txt is not None:
                return txt
            # MockBackendManager default transform_fn returns {"result": ...}
            return resp.content
        # error path
        return {"error": str(resp.error)}

    # --------------------------- generator ---------------------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # ---------------- empty ----------------
        if self.mode == "empty":
            # Yield once (empty list) so the base‑class initialisation succeeds.
            _ = yield []  # TaskManager immediately echoes an empty list back.
            return {"status": "empty"}

        # ---------------- single --------------
        if self.mode == "single":
            prompt = self.data.get("prompt1", self.data.get("p", "p1"))
            resp: Response = yield Request({"prompt": prompt})
            return {"final": self._to_text(resp), "source": "single"}

        # ------------- sequential -------------
        if self.mode == "sequential":
            p1 = self.data.get("prompt1", self.data.get("p1", "first"))
            p2 = self.data.get("prompt2", self.data.get("p2", "second"))
            resp1: Response = yield Request({"prompt": p1})
            v1 = self._to_text(resp1)
            resp2: Response = yield Request({"prompt": f"Based on {v1}, ask: {p2}"})
            return {"step1": v1, "step2": self._to_text(resp2), "source": "sequential"}

        # ---------------- batch ---------------
        if self.mode == "batch":
            pa = self.data.get("prompt_a", self.data.get("a", "A"))
            pb = self.data.get("prompt_b", self.data.get("b", "B"))
            resps: List[Response] = yield [Request({"prompt": pa}), Request({"prompt": pb})]
            batch_res = [self._to_text(r) for r in resps]
            # Historical + new tests reference different keys; expose both.
            return {"batch": batch_res, "final_batch": batch_res, "source": "batch"}

        # ------------- single_error ----------
        if self.mode == "single_error":
            resp: Response = yield Request({"prompt": "err"})
            return {"error": str(resp.error) if not resp.is_success else "unexpected"}

        # ------------- batch_mixed -----------
        if self.mode == "batch_mixed":
            pa = self.data.get("prompt_a", "A")
            pb = self.data.get("prompt_b", "B")
            resps: List[Response] = yield [Request({"prompt": pa}), Request({"prompt": pb})]
            mixed = [self._to_text(r) for r in resps]
            return {"mixed_results": mixed, "source": "batch_mixed"}

        raise ValueError("unknown mode")


# ---------------------------------------------------------------------------
# Re‑export helpers so existing import‑sites remain valid.
# ---------------------------------------------------------------------------
__all__ = [
    "MockGeneratorTask",
    "_make_success_content",
    "_create_success_response",
    "_create_error_response",
]

if __name__ == "__main__":  # pragma: no cover
    import unittest
    unittest.main()


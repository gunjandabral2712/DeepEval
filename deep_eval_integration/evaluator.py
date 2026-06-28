import os
from typing import Iterable

from deepeval.evaluate import evaluate
from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase


class DeepEvalEvaluator:
    """Adapter to run DeepEval evaluation on Python test cases."""

    def __init__(self, identifier: str | None = None):
        self.identifier = identifier or os.getenv(
            "DEEPEVAL_RUN_ID", "deepeval-python-test-framework"
        )
        self.metric = ExactMatchMetric()

    def build_test_case(
        self,
        name: str,
        input_text: str,
        actual_output: object,
        expected_output: object,
    ) -> LLMTestCase:
        return LLMTestCase(
            input=input_text,
            actual_output=str(actual_output),
            expected_output=str(expected_output),
            name=name,
            multimodal=False,
        )

    def evaluate_cases(self, test_cases: Iterable[LLMTestCase]):
        """Evaluate a list of DeepEval test cases and return the evaluation result."""
        return evaluate(
            test_cases=list(test_cases),
            metrics=[self.metric],
            identifier=self.identifier,
        )

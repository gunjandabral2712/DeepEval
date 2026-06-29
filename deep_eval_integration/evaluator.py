import os
from typing import Iterable, Optional

from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import DisplayConfig
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

    def evaluate_cases(
        self,
        test_cases: Iterable[LLMTestCase],
        metrics: list | None = None,
        display_config: Optional[DisplayConfig] = None,
    ):
        """Evaluate a list of DeepEval test cases and return the evaluation result.

        Args:
            test_cases: iterable of `LLMTestCase` objects.
            metrics: optional list of DeepEval metric instances. If None, uses
                `ExactMatchMetric` by default.
            display_config: optional `DisplayConfig` to control output verbosity,
                report file export location (``results_folder``), etc.
        """
        metrics = metrics or [self.metric]
        kwargs = dict(
            test_cases=list(test_cases),
            metrics=metrics,
            identifier=self.identifier,
        )
        if display_config is not None:
            kwargs["display_config"] = display_config
        return evaluate(**kwargs)

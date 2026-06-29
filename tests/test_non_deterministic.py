"""
Non-deterministic output evaluation using DeepEval.

Concepts demonstrated:
─────────────────────────────────────────────────────────────────────────────
• Temperature  – a float (0.0–1.0) that controls how "creative" (and how
                 likely to hallucinate) the model output is.
                 temperature=0.0 → grounded, deterministic output
                 temperature=1.0 → random, likely-to-hallucinate output

• Hallucination – the model claims something that contradicts the supplied
                  context.  Measured by HallucinationMetric (lower score =
                  fewer hallucinations = better).

• Answer Relevancy – how relevant the answer is to the question.
                     Measured by AnswerRelevancyMetric (higher score =
                     more relevant = better).

Each metric has a threshold that determines pass/fail:
  HallucinationMetric:   score <= threshold → PASS  (hallucination rate is low)
  AnswerRelevancyMetric: score >= threshold → PASS  (answer is relevant enough)

TemperatureLLM simulates both behaviours without making real API calls.
─────────────────────────────────────────────────────────────────────────────
"""

import json
from pathlib import Path

from deepeval.evaluate.configs import DisplayConfig
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

# DeepEval JSON reports land here (created by conftest.py before tests run)
DEEPEVAL_REPORT_DIR = str(
    Path(__file__).parent.parent / "reports" / "deepeval")


# ─────────────────────────────────────────────────────────────────────────────
# TemperatureLLM – stub that simulates deterministic vs hallucinatory outputs
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureLLM(DeepEvalBaseLLM):
    """
    A stub LLM that mimics the effect of a temperature parameter.

    - temperature=0.0  → always returns context-grounded, factual verdicts
                          (no hallucination, high relevancy)
    - temperature=1.0  → always returns fabricated, off-topic verdicts
                          (full hallucination, zero relevancy)
    - 0 < temperature < 1 → linear mix: fraction of hallucinated verdicts
                             proportional to temperature
    """

    def __init__(self, temperature: float = 0.0):
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        self.temperature = temperature

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return f"TemperatureLLM(t={self.temperature})"

    # ── Schema-aware response builder ────────────────────────────────────────

    def generate_with_schema(self, *args, **kwargs):
        """
        Return JSON strings that satisfy DeepEval's internal metric schemas.

        Schema routing is done by checking schema.__name__ and __module__:
          hallucination.Verdicts → verdict "yes" = aligns (good), "no" = hallucinated
          answer_relevancy.Verdicts → verdict "yes" = relevant, "no" = irrelevant
          *ScoreReason / Statements → simple helper responses
        """
        # DeepEval calls generate_with_schema(prompt, schema=SchemaClass)
        schema = kwargs.get("schema") or (args[1] if len(args) > 1 else None)
        name = getattr(schema, "__name__", "")
        module = getattr(schema, "__module__", "")

        # ── HallucinationMetric schemas ───────────────────────────────────────
        # DeepEval convention: verdict="yes" → output ALIGNS with context (good)
        #                       verdict="no"  → output CONTRADICTS context (hallucination)
        if "hallucination" in module and name == "Verdicts":
            # Low temperature  → output is grounded → verdict="yes" (aligns)
            # High temperature → output fabricates → verdict="no"  (contradicts)
            verdict = "yes" if self.temperature < 0.5 else "no"
            reason = (
                "The output is consistent with the provided context."
                if verdict == "yes"
                else "The output contradicts or fabricates facts not in the context."
            )
            return json.dumps(
                {"verdicts": [{"verdict": verdict, "reason": reason}]}
            )

        if name == "HallucinationScoreReason":
            if self.temperature >= 0.5:
                return json.dumps(
                    {"reason": f"High temperature ({self.temperature}) caused the "
                     "model to generate facts not present in the context."}
                )
            return json.dumps(
                {"reason": f"Low temperature ({self.temperature}) kept the "
                 "model grounded in the supplied context."}
            )

        # ── AnswerRelevancyMetric schemas ─────────────────────────────────────
        if name == "Statements":
            statements = (
                ["The answer directly addresses the question using context."]
                if self.temperature < 0.5
                else ["The answer drifts off topic due to high randomness."]
            )
            return json.dumps({"statements": statements})

        if "answer_relevancy" in module and name == "Verdicts":
            # At low temperature the answer is relevant ("yes").
            # At high temperature the answer is irrelevant ("no").
            verdict = "yes" if self.temperature < 0.5 else "no"
            reason = (
                "The statement directly answers the question."
                if verdict == "yes"
                else "The statement is unrelated to the original question."
            )
            return json.dumps(
                {"verdicts": [{"verdict": verdict, "reason": reason}]}
            )

        if name == "AnswerRelevancyScoreReason":
            return json.dumps(
                {"reason": "Score based on simulated temperature output."}
            )

        # Fallback
        return json.dumps({})

    async def a_generate_with_schema(self, *args, **kwargs):
        return self.generate_with_schema(*args, **kwargs)

    def generate(self, *args, **kwargs) -> str:
        return "{}"

    async def a_generate(self, *args, **kwargs) -> str:
        return self.generate(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Shared test fixture data
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT = [
    "The Eiffel Tower is located in Paris, France.",
    "It was constructed between 1887 and 1889 by Gustave Eiffel.",
    "It stands 330 metres tall and was the world's tallest structure until 1930.",
]

QUESTION = "Where is the Eiffel Tower, who built it, and when?"

# Low-temperature output – factual, grounded in context
GROUNDED_ANSWER = (
    "The Eiffel Tower is in Paris, France. "
    "It was built by Gustave Eiffel and completed in 1889."
)

# High-temperature output – plausible-sounding but contains fabrications
HALLUCINATED_ANSWER = (
    "The Eiffel Tower is in London, UK. "
    "It was built by Napoleon Bonaparte in 1756 as a naval beacon."
)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHallucinationMetric:
    """
    HallucinationMetric measures the fraction of context chunks that the
    model contradicts.

    Score range: 0.0 (no hallucination) → 1.0 (fully hallucinated)
    Pass condition: score <= threshold  (lower is better)
    """

    # ── Low temperature: grounded output should PASS ──────────────────────────
    def test_low_temperature_passes_hallucination_threshold(self):
        """
        temperature=0.0 → model stays grounded → score ≈ 0.0 → PASS.
        """
        low_temp_model = TemperatureLLM(temperature=0.0)

        test_case = LLMTestCase(
            name="low-temp-hallucination",
            input=QUESTION,
            actual_output=GROUNDED_ANSWER,
            context=CONTEXT,
        )

        # Lower threshold means stricter: only accept ≤ 50 % hallucination rate
        metric = HallucinationMetric(
            threshold=0.5,
            model=low_temp_model,
            async_mode=False,
        )
        metric.measure(test_case)

        assert metric.score == 0.0, (
            f"Expected hallucination score 0.0 for grounded output, got {metric.score}"
        )
        assert metric.is_successful(), (
            f"Expected low-temperature grounded output to PASS (score {metric.score} "
            f"should be ≤ threshold {metric.threshold})"
        )

    # ── High temperature: hallucinated output should FAIL ─────────────────────
    def test_high_temperature_fails_hallucination_threshold(self):
        """
        temperature=1.0 → model fabricates facts → score ≈ 1.0 → FAIL.
        """
        high_temp_model = TemperatureLLM(temperature=1.0)

        test_case = LLMTestCase(
            name="high-temp-hallucination",
            input=QUESTION,
            actual_output=HALLUCINATED_ANSWER,
            context=CONTEXT,
        )

        metric = HallucinationMetric(
            threshold=0.5,
            model=high_temp_model,
            async_mode=False,
        )
        metric.measure(test_case)

        assert metric.score == 1.0, (
            f"Expected hallucination score 1.0 for fabricated output, got {metric.score}"
        )
        assert not metric.is_successful(), (
            f"Expected high-temperature hallucinated output to FAIL (score {metric.score} "
            f"should be > threshold {metric.threshold})"
        )

    # ── Score comparison: high temp must score worse than low temp ─────────────
    def test_high_temperature_scores_worse_than_low_temperature(self):
        """
        Hallucination score must increase as temperature rises.
        """
        low_case = LLMTestCase(
            name="compare-low", input=QUESTION,
            actual_output=GROUNDED_ANSWER, context=CONTEXT,
        )
        high_case = LLMTestCase(
            name="compare-high", input=QUESTION,
            actual_output=HALLUCINATED_ANSWER, context=CONTEXT,
        )

        low_metric = HallucinationMetric(
            threshold=0.5, model=TemperatureLLM(0.0), async_mode=False
        )
        high_metric = HallucinationMetric(
            threshold=0.5, model=TemperatureLLM(1.0), async_mode=False
        )

        low_metric.measure(low_case)
        high_metric.measure(high_case)

        assert high_metric.score > low_metric.score, (
            f"High-temperature score ({high_metric.score}) should exceed "
            f"low-temperature score ({low_metric.score})"
        )


class TestAnswerRelevancyMetric:
    """
    AnswerRelevancyMetric measures how relevant the actual output is to the
    original question.

    Score range: 0.0 (irrelevant) → 1.0 (fully relevant)
    Pass condition: score >= threshold  (higher is better)
    """

    def test_low_temperature_passes_relevancy_threshold(self):
        """
        temperature=0.0 → answer stays on topic → score ≈ 1.0 → PASS.
        """
        test_case = LLMTestCase(
            name="low-temp-relevancy",
            input=QUESTION,
            actual_output=GROUNDED_ANSWER,
        )
        metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=TemperatureLLM(0.0),
            async_mode=False,
        )
        metric.measure(test_case)

        assert metric.score >= 0.7, (
            f"Expected relevancy ≥ 0.7 for on-topic output, got {metric.score}"
        )
        assert metric.is_successful()

    def test_high_temperature_fails_relevancy_threshold(self):
        """
        temperature=1.0 → answer drifts off topic → score ≈ 0.0 → FAIL.
        """
        test_case = LLMTestCase(
            name="high-temp-relevancy",
            input=QUESTION,
            actual_output=HALLUCINATED_ANSWER,
        )
        metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=TemperatureLLM(1.0),
            async_mode=False,
        )
        metric.measure(test_case)

        assert metric.score < 0.7, (
            f"Expected relevancy < 0.7 for off-topic output, got {metric.score}"
        )
        assert not metric.is_successful()


class TestCombinedMetrics:
    """
    Run both HallucinationMetric and AnswerRelevancyMetric through
    DeepEvalEvaluator.evaluate_cases() in a single pass.

    Both tests pass a DisplayConfig that:
      - saves a full JSON test-run report to reports/deepeval/
      - enables verbose_mode so per-verdict reasoning is printed
      - disables truncation so every result (pass and fail) is visible

    The JSON files are picked up by:
      - conftest.pytest_sessionfinish  → terminal summary after the suite
      - CI upload-artifact step        → downloadable from the GitHub Actions run
    """

    @staticmethod
    def _report_config(subfolder: str) -> DisplayConfig:
        return DisplayConfig(
            results_folder=DEEPEVAL_REPORT_DIR,
            results_subfolder=subfolder,
            verbose_mode=True,
            print_results=True,
            truncate_passing_cases=False,
            inspect_after_run=False,
        )

    def test_low_temperature_passes_all_metrics(self):
        from deep_eval_integration.evaluator import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator(identifier="temp-combined-low")
        model = TemperatureLLM(temperature=0.0)

        case = LLMTestCase(
            name="combined-low-temp",
            input=QUESTION,
            actual_output=GROUNDED_ANSWER,
            context=CONTEXT,
        )

        result = evaluator.evaluate_cases(
            [case],
            metrics=[
                HallucinationMetric(threshold=0.5, model=model,
                                    async_mode=False, verbose_mode=True),
                AnswerRelevancyMetric(
                    threshold=0.7, model=model, async_mode=False, verbose_mode=True),
            ],
            display_config=self._report_config("low-temperature"),
        )

        tr = result.test_results[0]
        assert tr.success, (
            "Expected low-temperature case to pass both metrics. "
            f"Metric results: {[(m.name, m.score, m.success) for m in tr.metrics_data]}"
        )

    def test_high_temperature_fails_all_metrics(self):
        from deep_eval_integration.evaluator import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator(identifier="temp-combined-high")
        model = TemperatureLLM(temperature=1.0)

        case = LLMTestCase(
            name="combined-high-temp",
            input=QUESTION,
            actual_output=HALLUCINATED_ANSWER,
            context=CONTEXT,
        )

        result = evaluator.evaluate_cases(
            [case],
            metrics=[
                HallucinationMetric(threshold=0.5, model=model,
                                    async_mode=False, verbose_mode=True),
                AnswerRelevancyMetric(
                    threshold=0.7, model=model, async_mode=False, verbose_mode=True),
            ],
            display_config=self._report_config("high-temperature"),
        )

        tr = result.test_results[0]
        assert not tr.success, (
            "Expected high-temperature case to fail at least one metric. "
            f"Metric results: {[(m.name, m.score, m.success) for m in tr.metrics_data]}"
        )

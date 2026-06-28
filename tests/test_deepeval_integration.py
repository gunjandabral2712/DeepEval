from deep_eval_integration.evaluator import DeepEvalEvaluator
from app.calculator import add, divide, is_prime
from deepeval.metrics import ExactMatchMetric, AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
import json


class DummyModel(DeepEvalBaseLLM):
    def load_model(self):
        return self

    def get_model_name(self):
        return "dummy"

    def generate_with_schema(self, prompt, schema=None):
        name = getattr(schema, '__name__', '')
        if name == 'Statements':
            return json.dumps({"statements": ["placeholder"]})
        if name == 'Verdicts':
            return json.dumps({"verdicts": [{"verdict": "yes", "reason": "relevant"}]})
        if name == 'AnswerRelevancyScoreReason':
            return json.dumps({"reason": "relevant"})
        return json.dumps({})

    async def a_generate_with_schema(self, prompt, schema=None):
        return self.generate_with_schema(prompt, schema=schema)

    # Implement abstract generate methods to satisfy DeepEvalBaseLLM
    def generate(self, *args, **kwargs):
        # simple string response
        return "{}"

    async def a_generate(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


def test_deepeval_metrics_evaluator():
    evaluator = DeepEvalEvaluator()
    cases = [
        evaluator.build_test_case("addition", "add(2, 3)", add(2, 3), 5),
        evaluator.build_test_case(
            "division", "divide(10, 2)", divide(10, 2), 5.0),
        evaluator.build_test_case(
            "prime-check", "is_prime(7)", is_prime(7), True),
    ]

    relevancy = AnswerRelevancyMetric(
        threshold=0.7, model=DummyModel(), async_mode=False)
    exact = ExactMatchMetric()

    result = evaluator.evaluate_cases(cases, metrics=[exact, relevancy])

    assert len(result.test_results) == 3
    assert all(test_result.success for test_result in result.test_results)

# DeepEval Python Test Framework

This repository demonstrates a practical integration between application unit tests and DeepEval-driven evaluation metrics. It shows how to keep application code, DeepEval evaluation adapters, and test cases clearly separated so teams can run fast unit tests while also exercising richer evaluation metrics.

**Key ideas**
- Application code lives under `app/` and is agnostic to the test framework.
- DeepEval adapters and helpers live under `deep_eval_integration/` and are used only by tests.
- `tests/` contains both standard `pytest` unit tests and DeepEval integration tests so you can run either or both.

## Repository structure

- `app/` — application code under test (example: `calculator.py`).
- `deep_eval_integration/` — a small adapter (`DeepEvalEvaluator`) that builds `LLMTestCase` instances and calls `deepeval.evaluate()`.
- `tests/` — `pytest` test files, including an integration test which demonstrates using multiple DeepEval metrics.
- `.github/workflows/ci.yml` — CI workflow that runs `pytest` on push/PRs.

## Installation

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running tests

Run the full test suite with `pytest`:

```bash
python -m pytest
```

You can also run a single test file:

```bash
python -m pytest tests/test_deepeval_integration.py -q
```

## DeepEval integration overview

The adapter `DeepEvalEvaluator` (in `deep_eval_integration/evaluator.py`) exposes two main helpers:

- `build_test_case(name, input_text, actual_output, expected_output)` — builds `deepeval.test_case.LLMTestCase` objects from local inputs/outputs.
- `evaluate_cases(test_cases, metrics=None)` — runs `deepeval.evaluate()` for the provided test cases and metric instances.

Example (from tests):

- Exact match metric (strict equality): `ExactMatchMetric()`
- Answer relevancy metric (semantic relevance, threshold 0.0–1.0): `AnswerRelevancyMetric(threshold=0.7)`

In the integration test we demonstrate using both metrics together so you can combine exact string checks with semantic checks.

## Avoiding external API keys in CI / local tests

Some DeepEval metrics use an LLM under the hood (OpenAI, Anthropic, etc.). To make the test suite runnable without external API keys, the integration test provides a small `DummyModel` shim implementing the minimal `DeepEvalBaseLLM` interface. This lets the `AnswerRelevancyMetric` return deterministic, structured responses during tests.

If you want to run metrics against a real LLM, configure provider keys in your environment (examples):

- `OPENAI_API_KEY` for OpenAI models
- `GOOGLE_API_KEY` / provider-specific keys for other providers

Or pass an explicit model instance to `AnswerRelevancyMetric(model=...)` when constructing the metric.

## Example usage (from tests)

1. Build test cases from application outputs:

```py
from deep_eval_integration.evaluator import DeepEvalEvaluator
from app.calculator import add

e = DeepEvalEvaluator()
case = e.build_test_case('addition', 'add(2,3)', add(2,3), 5)
```

2. Define metrics and run evaluation:

```py
from deepeval.metrics import ExactMatchMetric, AnswerRelevancyMetric

exact = ExactMatchMetric()
relevancy = AnswerRelevancyMetric(threshold=0.7)  # 0.0–1.0
result = e.evaluate_cases([case], metrics=[exact, relevancy])
```

3. Inspect the evaluation result (scores, pass/fail, verbose logs): the returned object is a `deepeval.evaluate.types.EvaluationResult` containing `test_results` with per-metric `MetricData`.

## Continuous integration

The included GitHub Actions workflow `.github/workflows/ci.yml` installs dependencies and runs `pytest` for every push and PR on `main`.


## Extending the framework

- Add more DeepEval metrics from `deepeval.metrics` (e.g., `SummarizationMetric`, `HallucinationMetric`).
- Use `DeepEvalEvaluator.evaluate_cases(..., metrics=[...])` to mix metrics.
- Replace `DummyModel` with a real `DeepEvalBaseLLM` implementation when you want to evaluate against a real LLM.

## Notes & caveats

- Running metrics that call external LLMs will incur latency and may require API keys and costs.
- The `DummyModel` used in tests is only suitable for deterministic unit tests; it should not be used for production evaluations where a real LLM is required.

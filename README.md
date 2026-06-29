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

## Non-deterministic output testing

**File:** `tests/test_non_deterministic.py`

Real LLMs are non-deterministic — the same question can produce a factual answer one run and a hallucinated answer the next, depending on the model's *temperature* setting. This test file demonstrates how to evaluate and gate that behaviour using DeepEval metrics, without making any real API calls.

### Core concepts

| Concept | Description |
|---|---|
| **Temperature** | A float `0.0–1.0`. Low values → grounded, predictable output. High values → creative but likely to hallucinate. |
| **Hallucination** | The model asserts something that contradicts the supplied reference context. `HallucinationMetric` measures the fraction of context chunks the output contradicts. **Lower score = fewer hallucinations = better.** |
| **Answer Relevancy** | How on-topic the output is relative to the original question. `AnswerRelevancyMetric` measures the fraction of output statements that are relevant. **Higher score = more relevant = better.** |

### The `TemperatureLLM` stub — location and behaviour

`TemperatureLLM` is a local model shim defined at the top of `tests/test_non_deterministic.py`. It subclasses `deepeval.models.base_model.DeepEvalBaseLLM` and replaces all network calls with deterministic JSON responses, making the full metric pipeline testable offline.

**Constructor:**

```py
TemperatureLLM(temperature=0.0)  # 0.0 = fully grounded, 1.0 = fully hallucinated
```

**How `generate_with_schema` decides what to return:**

DeepEval metrics work by calling `model.generate_with_schema(prompt, schema=SchemaClass)` internally. `TemperatureLLM` inspects the `schema.__name__` and `schema.__module__` of every call and routes it to the correct JSON structure:

| Schema class | Called by | `temperature < 0.5` returns | `temperature >= 0.5` returns |
|---|---|---|---|
| `hallucination.Verdicts` | `HallucinationMetric` | `{"verdicts": [{"verdict": "yes", "reason": "consistent with context"}]}` | `{"verdicts": [{"verdict": "no", "reason": "contradicts context"}]}` |
| `HallucinationScoreReason` | `HallucinationMetric` | `{"reason": "Low temperature kept the model grounded…"}` | `{"reason": "High temperature caused fabrication…"}` |
| `answer_relevancy.Statements` | `AnswerRelevancyMetric` | `{"statements": ["The answer directly addresses the question."]}` | `{"statements": ["The answer drifts off topic."]}` |
| `answer_relevancy.Verdicts` | `AnswerRelevancyMetric` | `{"verdicts": [{"verdict": "yes", "reason": "directly answers the question"}]}` | `{"verdicts": [{"verdict": "no", "reason": "unrelated to the question"}]}` |
| `AnswerRelevancyScoreReason` | `AnswerRelevancyMetric` | `{"reason": "Score based on simulated temperature output."}` | same |

> **DeepEval verdict convention for `HallucinationMetric`:**
> `verdict="yes"` means the output *aligns* with that context chunk (good — counts as grounded).
> `verdict="no"` means the output *contradicts* the context chunk (bad — counts as a hallucination).
> This is the inverse of what you might expect, so `TemperatureLLM` handles it explicitly.

### Shared fixture data

The test data is defined as module-level constants so all test classes share the same scenario — a factual question about the Eiffel Tower:

```py
CONTEXT = [
    "The Eiffel Tower is located in Paris, France.",
    "It was constructed between 1887 and 1889 by Gustave Eiffel.",
    "It stands 330 metres tall and was the world's tallest structure until 1930.",
]
QUESTION = "Where is the Eiffel Tower, who built it, and when?"

# temperature=0 → this is the expected output
GROUNDED_ANSWER = "The Eiffel Tower is in Paris, France. It was built by Gustave Eiffel and completed in 1889."

# temperature=1 → deliberately wrong
HALLUCINATED_ANSWER = "The Eiffel Tower is in London, UK. It was built by Napoleon Bonaparte in 1756 as a naval beacon."
```

### Test classes and what they assert

#### `TestHallucinationMetric`

Threshold: `0.5` (pass if hallucination rate ≤ 50 %).

| Test | Temperature | Expected score | Expected result |
|---|---|---|---|
| `test_low_temperature_passes_hallucination_threshold` | `0.0` | `0.0` | ✅ PASS |
| `test_high_temperature_fails_hallucination_threshold` | `1.0` | `1.0` | ❌ FAIL |
| `test_high_temperature_scores_worse_than_low_temperature` | both | high > low | ordering check |

#### `TestAnswerRelevancyMetric`

Threshold: `0.7` (pass if relevancy ≥ 70 %).

| Test | Temperature | Expected score | Expected result |
|---|---|---|---|
| `test_low_temperature_passes_relevancy_threshold` | `0.0` | `1.0` | ✅ PASS |
| `test_high_temperature_fails_relevancy_threshold` | `1.0` | `0.0` | ❌ FAIL |

#### `TestCombinedMetrics`

Uses `DeepEvalEvaluator.evaluate_cases()` to run **both** metrics together in a single evaluation pass:

| Test | Temperature | Both metrics pass? |
|---|---|---|
| `test_low_temperature_passes_all_metrics` | `0.0` | ✅ yes |
| `test_high_temperature_fails_all_metrics` | `1.0` | ❌ no (at least one fails) |

### Run just the non-deterministic tests

```bash
python -m pytest tests/test_non_deterministic.py -v
```

---

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

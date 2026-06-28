# DeepEval Python Test Framework

This repository demonstrates a clear separation between the application code under test and the test framework.

## Structure

- `app/` - application code being tested.
- `deep_eval_integration/` - DeepEval evaluation adapter and integration code.
- `tests/` - pytest-based test suite.

## Goals

- Use DeepEval as the evaluation application when available.
- Keep application logic separate from test logic.
- Use `pytest` for standard unit testing and `deepeval` for evaluation metrics.

## Usage

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run unit tests:

```bash
python -m pytest
```

Run DeepEval evaluation via Python:

```bash
python -c "from deep_eval_integration.evaluator import DeepEvalEvaluator; from app.calculator import add, divide, is_prime; evaluator = DeepEvalEvaluator(); cases = [evaluator.build_test_case('addition', 'add(2,3)', add(2,3), 5), evaluator.build_test_case('division', 'divide(10,2)', divide(10,2), 5.0), evaluator.build_test_case('prime-check', 'is_prime(7)', is_prime(7), True)]; result = evaluator.evaluate_cases(cases); print(result)"
```

## Notes

- `deepeval` is used for evaluation metrics, while `pytest` is used for standard test assertions.
- The sample DeepEval integration uses `ExactMatchMetric` to compare actual and expected outputs.

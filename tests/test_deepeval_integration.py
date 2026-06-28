from deep_eval_integration.evaluator import DeepEvalEvaluator
from app.calculator import add, divide, is_prime


def test_deepeval_exact_match_evaluator():
    evaluator = DeepEvalEvaluator()
    cases = [
        evaluator.build_test_case("addition", "add(2, 3)", add(2, 3), 5),
        evaluator.build_test_case(
            "division", "divide(10, 2)", divide(10, 2), 5.0),
        evaluator.build_test_case(
            "prime-check", "is_prime(7)", is_prime(7), True),
    ]
    result = evaluator.evaluate_cases(cases)

    assert len(result.test_results) == 3
    assert all(test_result.success for test_result in result.test_results)

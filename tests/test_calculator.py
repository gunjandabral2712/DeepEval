from app.calculator import add, divide, is_prime
import pytest


def test_addition():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_division():
    assert divide(10, 2) == 5.0
    assert divide(9, 3) == 3.0


def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(5, 0)


def test_is_prime():
    assert is_prime(2)
    assert is_prime(17)
    assert not is_prime(1)
    assert not is_prime(15)

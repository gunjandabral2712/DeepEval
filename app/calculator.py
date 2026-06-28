def add(a, b):
    """Return the sum of two numbers."""
    return a + b


def divide(a, b):
    """Return the division of a by b, raising ZeroDivisionError when b is zero."""
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b


def is_prime(n):
    """Return True if n is a prime number."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

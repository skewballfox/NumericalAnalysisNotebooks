from typing import Callable, Tuple, List
import numpy as np


def bisect(
    f: Callable[[int | float], float],
    left_endpoint: int | float,
    right_endpoint: int | float,
    TOL: float,
    itmax: int,
) -> Tuple[float, float, float]:
    """
    performs a bisection method with the following stopping criterion:
        - relative error <= tolerance or
        - the max number of iterations is exceeded

    Parameters
    ----------
    f : the function being bisected
    left_endpoint: a, the start of the continuous domain
    right_endpoint: b, the end of the continuous domain
    TOL: the error tolerance
    itmax: the max number of iterations

    Returns
    -------
    pivot: p, the approximated root of the function
    err: the error estimate for res
    range_pivot: the output of f(pivot)
    """

    range_left_b = f(left_endpoint)
    range_right_b = f(right_endpoint)

    assert (
        range_left_b * range_right_b < 0
    ), f"bisection only works if f(a) and f(b) are on opposite sides of the x axis"
    pivot: float

    for k in (1, itmax):
        pivot = (left_endpoint + right_endpoint) / 2.0
        range_pivot = f(pivot)
        if range_pivot == 0:
            left_endpoint = right_endpoint = pivot
        elif range_left_b * range_pivot < 0:
            right_endpoint = pivot
            range_right_b = range_pivot
        else:
            left_endpoint = pivot
            range_left_b = range_pivot

        if right_endpoint - left_endpoint < TOL:
            break

    pivot = (left_endpoint + right_endpoint) / 2.0
    err = (right_endpoint - left_endpoint) / 2.0
    range_pivot = f(pivot)

    return (pivot, err, range_pivot)


def fixed_iter(
    g: Callable[[int | float], float], x: int | float, iter_count: int
) -> List[float]:
    result = []
    p: float = x
    for i in range(0, iter_count):
        p = g(p)
        result.append(p)
    return result


def _newton_horner(coefficients, x0):
    d = 0
    p = coefficients[-1]
    for coef in coefficients[-2::-1]:

        d = p + (d * x0)
        p = coef + (x0 * p)
    return (p, d)


# TODO: something is messed up either with this or it's helper function
def newton_horner(coefficients, x0, TOL, iter_max):
    x = x0
    h: float
    for i in range(1, iter_max):
        [p, d] = _newton_horner(coefficients, x)
        h = -(p / d)
        x = x + h
        if abs(h) < TOL:
            return x
    return x


def bairstow(coefficients, u0, v0, TOL, itmax):
    n = len(coefficients)
    u = u0
    v = v0
    b = np.zeros(n)
    c = np.zeros(n)
    b[-1] = c[-2] = coefficients[-1]
    c[-1] = 0
    DetJ: float
    for i in range(itmax):
        b[-2] = coefficients[-2] + u * b[-1]
        for j in range(n - 3, 0, -1):
            b[j] = coefficients[j] + u * b[j + 1] + v * b[j + 2]
            c[j] = b[j + 1] + u * c[j + 1] + v * c[j + 2]
    DetJ = c[0] * c[2] - c[1] * c[1]
    du = (c[1] * b[1] - c[2] * b[0]) / DetJ
    dv = (c[1] * b[0] - c[0] * b[1]) / DetJ
    u += du
    v += dv
    return (u, v)

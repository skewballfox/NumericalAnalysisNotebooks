from typing import OrderedDict
from typing import Callable, Tuple, List

import numpy as np


def bisect_with_table(
    f: Callable[[int | float], float],
    left_endpoint: int | float,
    right_endpoint: int | float,
    TOL: float,
    itmax: int,
    p: int | float,
) -> Tuple[float, float, float]:
    """
    performs a bisection method with the following stopping criterion:
        - relative error <= tolerance or
        - the max number of iterations is exceeded

    Parameters
    ----------
    f : the function being bisected
    left_endpoint: a, the start of the continuous range
    right_endpoint: b, the end of the continuous range
    TOL: the error tolerance
    itmax: the max number of iterations
    p : the real p that is being approximated

    Returns
    ---
    p: p, the approximated root of the function
    err: the error estimate for res
    range_pivot: the output of f(pivot)
    """

    range_left_b = f(left_endpoint)
    range_right_b = f(right_endpoint)

    assert (
        range_left_b * range_right_b < 0
    ), f"bisection only works if f(a) and f(b) are on opposite sides of the x axis"
    pivot: float
    results: OrderedDict = {"iter": [], "p_n": [], "|p-p_n|": [], "|p-p_n-1|": []}
    for k in range(1, itmax + 1):
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

        results["iter"].append(k)
        results["p_n"].append(pivot)
        results["|p-p_n|"].append(abs(p - pivot))
        results["|p-p_n-1|"].append(0 if k == 1 else results["|p-p_n|"][-2])

        if right_endpoint - left_endpoint < TOL:
            break

    pivot = (left_endpoint + right_endpoint) / 2.0
    err = (right_endpoint - left_endpoint) / 2.0
    range_pivot = f(pivot)

    return results


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

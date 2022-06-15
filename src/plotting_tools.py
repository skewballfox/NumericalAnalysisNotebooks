import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def centered_plot(
    f: Callable[[int | float], float], left_b: int | float, right_b: int | float
):
    xvals = np.linspace(left_b, right_b, num=100)
    yvals = list((map(f, xvals)))
    # use set_position
    ax = plt.gca()
    # ax.spines["top"].set_color("none")
    # ax.spines["left"].set_position("zero")
    # ax.spines["right"].set_color("none")
    # ax.spines["bottom"].set_position("zero")
    plt.xlim(xvals[0], xvals[-1])
    plt.ylim(np.min(yvals), np.max(yvals))
    plt.plot(xvals, yvals)
    plt.grid(True)
    plt.show()

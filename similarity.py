"""
A set of vectorized functions for measuring the similarity between two
identically sized arrays.
"""
import numpy as np


def cummean(x, axis=0):
    """
    Returns the cumulative means for <x> along <axis>.
    """

    x = np.array(x)

    # N, number of samples, increases with each element.
    N = np.arange(1, x.shape[axis] + 1)

    # The cumulatve mean is the sum divided by N.
    return np.cumsum(x, axis) / N


def l2(x1, x2, axis=0):
    """
    Returns the 2d euclidian distance (L2) between x1 and x2.
    """

    x1 = np.array(x1)
    x2 = np.array(x2)
    distances = np.sqrt(x1 ** 2 + x2 ** 2)

    return distances


def l2_cummean(x, y, axis=0):
    """
    For each entry i (along <axis>), and each x_i and y_i pair return the
    distance from that point to the point representing the cumulative mean
    of <x> and <y> from i_initial to i.
    """

    x = np.array(x)
    y = np.array(y)

    x_cmean = cummean(x, axis)
    y_cmean = cummean(y, axis)

    return l2(y - y_cmean, x - x_cmean, axis)

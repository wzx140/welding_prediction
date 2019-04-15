import numpy as np


def abs_dist(x: np.ndarray, y: np.ndarray):
    """
    distance functions for point-point distances
    :param x: nd array, T1 x D
    :param y: nd array, T2 x D
    :return: matrix with shape of (T1, T2)
    """
    return np.sum(np.abs(x[:, None, :] - y[None, :, :]), axis=2)


def sq_dist(x: np.ndarray, y: np.ndarray):
    """
    distance functions for point-point distances
    :param x: nd array, T1 x D
    :param y: nd array, T2 x D
    :return: matrix with shape of (T1, T2)
    """
    return np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)


def dtw(x: np.ndarray, y: np.ndarray, dist=sq_dist):
    """
    Main dynamic time warping function, computes dist of two sequences
    :param x: nd array, T1 x D
    :param y: nd array, T2 x D
    :param func dist: distance used as cost measure (default L1 norm)
    :returns the minimum distance, the accumulated cost matrix and the wrap path
    """
    x = np.array(x)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    y = np.array(y)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # store length of each sequence
    r, c = x.shape[0], y.shape[0]

    D = np.zeros((r + 1, c + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf

    # handle pairwise dist calculation with broadcasting
    D[1:, 1:] = dist(x, y)

    # sum up distance from neighboring cells
    for i in range(r):
        for j in range(c):
            D[i + 1, j + 1] += min(D[i, j], D[i, j + 1], D[i + 1, j])

    D = D[1:, 1:]
    dist = D[-1, -1] / sum(D.shape)
    return dist, D, _trackeback(D)


def _trackeback(D: np.ndarray):
    """
    track the path of the DTW
    :param D: nd array, T1 x D
    :return the wrap path
    """
    i, j = np.array(D.shape) - 1
    p, q = [i], [j]
    while i > 0 and j > 0:
        tb = np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]))

        if tb == 0:
            i = i - 1
            j = j - 1
        elif tb == 1:
            i = i - 1
        elif tb == 2:
            j = j - 1

        p.insert(0, i)
        q.insert(0, j)

    if p[0] == 0:
        while j > 0:
            j -= 1
            p.insert(0, i)
            q.insert(0, j)
    elif q[0] == 0:
        while i > 0:
            i -= 1
            p.insert(0, i)
            q.insert(0, j)

    return np.array(p), np.array(q)

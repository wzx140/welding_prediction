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


def dtw(x: np.ndarray, y: np.ndarray, dist=abs_dist):
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
    dist = D[-1, -1]
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


def norm_length(p_list: list, q_list: list):
    """
    get equal length

    :param p_list: the corresponding index of first data list
    :param q_list: the corresponding index of 2-2050 data list
    :return: the index of result of matrix
    """
    assert len(p_list) == len(q_list)

    # the number of p_list or q_list
    num = len(p_list)
    # the number of indexes in one sample
    num_index = p_list[0][-1] + 1
    # the number list of each index in p in standard mode
    num_list_p_standard = [0] * num_index
    # the number list of each index in p_list
    num_list_p_list = []

    # extend p in a standard mode and store the number list of each index in p_list
    for p in p_list:
        temp_index = 0
        temp_num = 1
        # the number list of each index in p
        num_list_p = [1] * num_index
        for i in range(len(p)):
            if i == len(p) - 1:
                num_list_p[temp_index] = temp_num
                if num_list_p_standard[temp_index] < temp_num:
                    num_list_p_standard[temp_index] = temp_num

            elif p[i] == p[i + 1]:
                temp_num += 1
            elif num_list_p_standard[temp_index] < temp_num:
                num_list_p_standard[temp_index] = temp_num

                num_list_p[temp_index] = temp_num

                temp_index += 1
                temp_num = 1
            else:
                num_list_p[temp_index] = temp_num

                temp_index += 1
                temp_num = 1

        num_list_p_list.append(num_list_p.copy())

    assert len(num_list_p_list) == len(q_list)

    p_standard = []
    for i, _num in enumerate(num_list_p_standard):
        p_standard += [i] * _num

    # the output consists of index of data by column
    output = np.zeros((len(p_standard), num + 1), dtype=np.int)
    output[..., 0] = p_standard

    # extend q_list in a standard mode as p
    for i, num_list_p_item in enumerate(num_list_p_list):
        temp_index = 0
        temp_index_in_output = 0
        for j, num_index in enumerate(num_list_p_item):
            assert num_index <= num_list_p_standard[j]
            # extend the q
            result = extend(q_list[i][temp_index: temp_index + num_index], num_list_p_standard[j])
            assert num_list_p_standard[j] == len(result)
            assert result[0] == q_list[i][temp_index]
            assert result[-1] == q_list[i][temp_index + num_index - 1]
            temp_index += num_index

            # store in output
            output[temp_index_in_output:temp_index_in_output + len(result), i + 1] = result
            temp_index_in_output += len(result)

    return output


def extend(p: list, length: int):
    """
    extend the list to expected length

    :param length: the expected length, must bigger than length of p
    :param p: the input list
    :return:
    """
    delta = length - len(p)
    # the multiple of each index
    index_multiple_list = [1] * len(p)

    ####################################
    # two-way extend, eg:              #
    # input: p = 123345, length = 10   #
    # output 1122334455                #
    ####################################
    # the length for extend remained, after remove the integer multiple part
    temp_length = delta
    if delta >= len(p):
        multiple = delta // len(p) + 1
        index_multiple_list = [item * multiple for item in index_multiple_list]
        temp_length = delta - (multiple - 1) * len(p)
    # whether it is a odd number, 1->odd, 0->even
    flag = temp_length % 2
    temp_index = -1
    for i in range(temp_length // 2):
        index_multiple_list[i] += 1
        index_multiple_list[len(p) - 1 - i] += 1
        temp_index = i
    if flag == 1 and temp_length != 0:
        index_multiple_list[temp_index + 1] += 1

    return _extend(p, index_multiple_list)


def _extend(p: list, index_multiple_list: list):
    """
    extend the p with index_multiple_list

    :param p:
    :param index_multiple_list:
    :return:
    """
    assert len(p) == len(index_multiple_list)
    result = []
    for i, index_multiple in enumerate(index_multiple_list):
        for j in range(index_multiple):
            result.append(p[i])
    return result

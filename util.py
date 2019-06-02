import csv
import os
import math
import numpy as np
from imblearn.over_sampling import ADASYN


def load_train_data(load_num_good: int = 2000):
    """
    load welding test and train data
    :param: load_num_good: 0->all
    :return: plain data, and number of good and bad
    """
    data = []
    num_good = 0
    num_bad = 0
    prefix_good = 'data/good/'
    prefix_bad = 'data/bad/'

    # load all of the bad sample
    for file in os.listdir(prefix_bad):
        data.append(load_data(prefix_bad + file))
        num_bad += 1

    # load good sample
    for file in os.listdir(prefix_good):
        data.append(load_data(prefix_good + file))
        num_good += 1
        if num_good == load_num_good - 1:
            num_good = num_good + 1
            break

    return data, num_good, num_bad


def load_data(path):
    """
    load the data from file
    :param path:
    :return:
    """
    with open(path) as f:
        data = []
        f_csv = csv.reader(f, delimiter='|')
        for i, row in enumerate(f_csv):
            data_temp = [row[1], row[2], row[3]]
            data.append(data_temp)
    return np.array(data, dtype=np.float)


def regularize(matrix: np.ndarray, axis: int = 0):
    """
    make the data range from 0~1

    :param matrix:
    :param axis: 0->vertical operation, 1->horizontal operation
    :return:
    :exception: the initial standard deviation in axis must not be 0
    """
    assert axis == 0 or axis == 1
    min_ = np.min(matrix, axis=axis, keepdims=True)
    max_ = np.max(matrix, axis=axis, keepdims=True)
    return (matrix - min_) / (max_ - min_)


def shuffle_data(data, label):
    """
    randomly disrupt data
    :param data:
    :param label:
    :return:
    """

    permutation = list(np.random.permutation(len(data)))
    shuffled_x = data[permutation, :]
    shuffled_y = label[permutation, :]
    return shuffled_x, shuffled_y


def random_mini_batches(x, y, mini_batch_size=64):
    """
    Creates a list of random minibatches from (x, y)

    Arguments:
    :param x: input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    :param y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    :param mini_batch_size: size of the mini-batches, integer

    :return: list of synchronous (mini_batch_x, mini_batch_y)
    """

    m = x.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation, :]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_x = shuffled_x[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_y = shuffled_y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_x = shuffled_x[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_y = shuffled_y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches


def resample(data: list, length: int = 0):
    """
    Downsample, the default length after sampling is the minimum length
    :param data:
    :param length: 0->default
    :return: result and length
    """
    result = []
    if length == 0:
        length = min([len(d) for d in data])

    for item in data:
        index = np.linspace(0, len(item), length, endpoint=False, dtype=np.int)
        result.append(item[index, :])

    return result, length


def reshape(data: list, length: int):
    """
    reshape to (m, nx, 1, 3)
    :param data:
    :param length: the length of the time series
    :return:
    """
    result = np.zeros((len(data), length, 3), dtype=np.float)
    for i, item in enumerate(data):
        result[i, :, :] = item
    return result


def expand(x: np.ndarray, y: np.ndarray):
    """
    enlarge the number of the minority class use ADASYN
    :param x:
    :param y:
    :return:
    """
    ada = ADASYN(sampling_strategy=1)
    x_res, y_res = ada.fit_resample(x.reshape(-1, 600 * 3), y.reshape(-1))
    return x_res.reshape(-1, 600, 3), y_res.reshape(-1, 1), len(x_res.reshape(-1, 600, 3)) - len(x)

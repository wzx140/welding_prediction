import math
import random
import definitions
import h5py
import numpy as np


def load_data(load_num_good: int = 2000):
    """
    load welding test and train data
    :param: load_num_good: 0->all
    :return: plain data, and number of good and bad
    """
    data = []
    num_good = 0
    num_bad = 0
    with h5py.File(definitions.ROOT_DIR + 'dataSets/data.h5') as f:
        # load good data
        lengths = f['GOOD/LEN'][:]
        count = 0
        for num, length in enumerate(lengths):
            data.append(f['GOOD/DATA'][count:count + length])
            num_good += 1
            count += length

        data = random.sample(data, load_num_good)
        num_good = load_num_good

        # load bad data
        lengths = f['BAD/LEN'][:]
        count = 0
        for length in lengths:
            data.append(f['BAD/DATA'][count:count + length])
            num_bad += 1
            count += length

    return data, num_good, num_bad


def regularize(matrix: np.ndarray, axis: int = 0):
    """
    make the data's standard deviation is 1 and mean is 0 through axis 0 or 1

    :param matrix:
    :param axis: 0->vertical operation, 1->horizontal operation
    :return:
    :exception: the initial standard deviation in axis must not be 0
    """
    assert axis == 0 or axis == 1
    mean = np.mean(matrix, axis=axis, keepdims=True)
    std = np.std(matrix, axis=axis, keepdims=True)
    return (matrix - mean) / std


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


def add_noise(data: np.ndarray, num: int, scale: float):
    """
    expand the data with noise
    :param data:
    :param num: the number of adding noise sample
    :param scale: the range of the noise
    :return:
    """
    size, width, depth = data.shape
    result = np.zeros(shape=(num, width, depth), dtype=np.float)
    index = 0
    for i in range(num):
        noise = (np.random.rand(width, depth) - 0.5) * scale
        result[i] = data[index] + noise
        index = index + 1 if index != size - 1 else 0
    return result


def transition(data: np.ndarray, num: int, scale: float):
    """
    move data up and down
    :param data:
    :param num: the number of sample after moving
    :param scale: the range of the moving
    :return:
    """
    size, width, depth = data.shape
    result = np.zeros(shape=(num, width, depth), dtype=np.float)
    index = 0
    for i in range(num):
        distance = (random.random() - 0.5) * scale
        result[i] = data[index] + distance
        index = index + 1 if index != size - 1 else 0
    return result

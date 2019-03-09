import h5py
import numpy as np
import random

# from matplotlib import pyplot as plt


def load_data(incomplete: bool):
    """
    load welding test and train data
    :param: incomplete mood
    :return: plain data, and number of good and bad
    """
    data = []
    num_good = 0
    num_bad = 0
    with h5py.File('dataSets/data.h5') as f:
        # load good data
        lengths = f['GOOD/LEN'][:]
        count = 0
        for length in lengths:
            data.append(f['GOOD/DATA'][count:count + length])
            num_good += 1
            count += length

        if incomplete:
            data = random.sample(data, 50)
            num_good = 50

        # load bad data
        lengths = f['BAD/LEN'][:]
        count = 0
        for length in lengths:
            data.append(f['BAD/DATA'][count:count + length])
            num_bad += 1
            count += length

    return data, num_good, num_bad


def shuffle_data(data, num_good: int, num_bad: int, radio: float = 0.3):
    """
    randomly disrupt data
    :param radio: radio of test
    :param num_bad: number of good example in data
    :param num_good: number of bad example in data
    :param data: good + bad data after pretreatment
    :return: train_x, train_y, test_x, test_y
    """
    num = num_good + num_bad
    num_test = int(num * radio)
    num_train = num - num_test
    data_y = np.array([1] * num_good + [0] * num_bad, dtype=np.float32).reshape((1, num))

    permutation = list(np.random.permutation(num))
    shuffled_x = data[:, permutation]
    shuffled_y = data_y[:, permutation]
    train_x = shuffled_x[:, 0:num_train]
    train_y = shuffled_y[:, 0:num_train]
    test_x = shuffled_x[:, num_train:num]
    test_y = shuffled_y[:, num_train:num]
    return train_x, train_y, test_x, test_y


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


def resample(data: np.ndarray, length: int = 0):
    """

    :param data:
    :param length:
    :return:
    """
    result = []
    if length == 0:
        length = min([len(d) for d in data])

    for item in data:
        index = np.linspace(0, len(item), length, endpoint=False, dtype=np.int)
        result.append(item[index, :])

    return result


# if __name__ == '__main__':
#     data, num1, num2 = load_data(False)
#     data2 = resample(data, 600)
#     plt.figure()
#     plt.subplot(121)
#     plt.plot(data[0][:, 0])
#
#     plt.subplot(122)
#     plt.plot(data2[0][:, 0])
#
#     plt.show()

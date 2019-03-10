import h5py
import numpy as np


# from matplotlib import pyplot as plt


def load_data(load_num_good: int = 2000):
    """
    load welding test and train data
    :param: load_num_good: 0->all
    :return: plain data, and number of good and bad
    """
    data = []
    num_good = 0
    num_bad = 0
    with h5py.File('dataSets/data.h5') as f:
        # load good data
        lengths = f['GOOD/LEN'][:]
        count = 0
        for num, length in enumerate(lengths):
            data.append(f['GOOD/DATA'][count:count + length])
            num_good += 1
            count += length
            if num == load_num_good - 1:
                break

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
    result = np.zeros((len(data), 1, length, 3), dtype=np.float)
    for i, item in enumerate(data):
        result[i, :, :, 0] = item[:, 0]
        result[i, :, :, 1] = item[:, 1]
        result[i, :, :, 2] = item[:, 2]
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

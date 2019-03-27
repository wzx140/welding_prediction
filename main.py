import matplotlib.pyplot as plt
import tensorflow as tf
# for debug
from tensorflow.python import debug as tf_debug
import numpy as np

import log
import util
from config import *
from cnn import Cnn

if __name__ == '__main__':
    log.log_info('program start')
    data, num_good, num_bad = util.load_data(num_data // 2)
    log.log_debug('Data loading completed')

    # resample
    data, length = util.resample(data, 600)
    data = util.reshape(data, length)
    good_data_origin = data[:num_good, :]
    bad_data_origin = data[num_good:, :]

    # extract bad data for test and train
    permutation = list(np.random.permutation(len(bad_data_origin)))
    shuffled_bad_data = bad_data_origin[permutation, :]
    test_bad_data = shuffled_bad_data[:int(num_bad * 0.3), :]
    train_bad_data_origin = shuffled_bad_data[int(num_bad * 0.3):, :]
    # extract corresponding good data for test and train
    permutation = list(np.random.permutation(len(good_data_origin)))
    shuffled_good_data = good_data_origin[permutation, :]
    test_good_data = shuffled_good_data[:len(test_bad_data), :]
    train_good_data = shuffled_good_data[len(test_bad_data):, :]

    assert len(test_bad_data) == len(test_good_data)
    # construct test data
    test_y = np.array([1.] * len(test_good_data) + [0.] * len(test_bad_data), dtype=np.float).reshape(
        (len(test_bad_data) + len(test_good_data), 1))
    test_x = np.vstack((test_good_data, test_bad_data))

    # expand the number of bad data for train
    add_noise = int((len(train_good_data) - len(train_bad_data_origin)) * add_noise_radio)
    transition = len(train_good_data) - add_noise - len(train_bad_data_origin)
    if add_noise and transition:
        result1 = util.add_noise(train_bad_data_origin, add_noise, 1)
        result2 = util.transition(train_bad_data_origin, transition, 2)
        train_bad_data = np.vstack((result1, result2, train_bad_data_origin))
    else:
        train_bad_data = train_bad_data_origin

    assert len(train_bad_data) == len(train_good_data)
    # construct train data
    train_y = np.array([1.] * len(train_good_data) + [0.] * len(train_bad_data), dtype=np.float).reshape(
        (len(train_bad_data) + len(train_good_data), 1))
    train_x = np.vstack((train_good_data, train_bad_data))

    # regularize
    for i in range(len(train_x)):
        train_x[i, :, 0] = util.regularize(train_x[i, :, 0])
        train_x[i, :, 1] = util.regularize(train_x[i, :, 1])
        train_x[i, :, 2] = util.regularize(train_x[i, :, 2])
    for i in range(len(test_x)):
        test_x[i, :, 0] = util.regularize(test_x[i, :, 0])
        test_x[i, :, 1] = util.regularize(test_x[i, :, 1])
        test_x[i, :, 2] = util.regularize(test_x[i, :, 2])

    # random
    train_x, train_y = util.shuffle_data(train_x, train_y)

    log.log_debug('prepare completed')
    log.log_info('convolution layers: ' + str(conv_layers))
    log.log_info('filters: ' + str(filters))
    log.log_info('full connected layers: ' + str(learning_rate))
    log.log_info('learning rate: ' + str(fc_layers))
    log.log_info('keep prob: ' + str(keep_prob))
    log.log_info('the number of expanding bad data: ' + str(add_noise + transition))
    log.log_info('mini batch size: ' + str(mini_batch_size))

    if mini_batch_size != 0:
        assert mini_batch_size <= len(train_x)

    cnn = Cnn(conv_layers, fc_layers, filters, learning_rate)
    (m, n_W0, n_C0) = train_x.shape
    n_y = train_y.shape[1]

    # log the cost and accuracy
    cost_log = []
    test_log = []
    train_log = []
    x = []

    # construction calculation graph
    cnn.initialize(n_W0, n_C0, n_y)
    cnn.forward()
    cost = cnn.cost()
    optimizer = cnn.get_optimizer(cost)
    predict, accuracy = cnn.predict()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        if enable_debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        for i in range(1, num_epochs + 1):
            if mini_batch_size != 0:
                num_mini_batches = int(m / mini_batch_size)
                mini_batches = util.random_mini_batches(train_x, train_y, mini_batch_size)

                cost_value = 0
                for mini_batch in mini_batches:
                    (mini_batch_x, mini_batch_y) = mini_batch
                    _, temp_cost = sess.run([optimizer, cost], feed_dict={cnn.x: mini_batch_x, cnn.y: mini_batch_y,
                                                                          cnn.keep_prob: keep_prob})
                    cost_value += temp_cost
                cost_value /= num_mini_batches
            else:
                _, cost_value = sess.run([optimizer, cost],
                                         feed_dict={cnn.x: train_x, cnn.y: train_y, cnn.keep_prob: keep_prob})

            # disable dropout
            train_accuracy, output = sess.run([accuracy, predict],
                                              feed_dict={cnn.x: train_x, cnn.y: train_y, cnn.keep_prob: 1})
            test_accuracy = sess.run(accuracy, feed_dict={cnn.x: test_x, cnn.y: test_y, cnn.keep_prob: 1})

            cost_log.append(cost_value)
            train_log.append(train_accuracy)
            test_log.append(test_accuracy)
            x.append(i)

            if print_detail and (i % 10 == 0 or i == 1):
                info = '\nIteration %d\n' % i + \
                       'Cost: %f\n' % cost_value + \
                       'Train accuracy: %f\n' % train_accuracy + \
                       'Test accuracy: %f' % test_accuracy
                log.log_info(info)

    plt.figure()
    plt.subplot(121)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.plot(x, cost_log)

    plt.subplot(122)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.plot(x, train_log, label='train accuracy')
    plt.plot(x, test_log, label='test accuracy')
    plt.legend()

    plt.show()

    log.log_info('program end')

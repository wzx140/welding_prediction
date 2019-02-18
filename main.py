import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
# for debug
from tensorflow.python import debug as tf_debug

import dtw
import log
import util
from config import *
from dnn import Dnn

if __name__ == '__main__':

    if incomplete_mode:
        log.log_debug('Data loading begin: incomplete mode')
        data, num_good, num_bad = util.load_data(True)
    else:
        log.log_debug('Data loading begin')
        data, num_good, num_bad = util.load_data(False)
    log.log_info('Data loading completed')

    if fast_mode and not incomplete_mode:
        with h5py.File('dataSets/data.h5') as f:
            data_processed = f['processed_data'][:]
        log.log_info('Fast mood: Data processing completed')
    elif fast_mode and incomplete_mode:
        with h5py.File('dataSets/data.h5') as f:
            data_processed = f['processed_data_in'][:]
        log.log_info('Fast mood: Data processing completed')
    else:
        # data dtw with data 1
        p_list = []
        q_list = []
        log.log_debug('DTW begin')
        for i, data_ in enumerate(data[1:]):
            dist, matrix, track = dtw.dtw(data[0], data_)
            # the mapping of the index
            p_list.append(track[0])
            q_list.append(track[1])
            log.log_debug('Data 1 and %d DTW completed' % (i + 2))
        log.log_info('DTW completed')

        # make the sequence equal length
        log.log_debug('Other processing begin')
        result = dtw.norm_length(p_list, q_list)
        for i in range(len(data)):
            data[i] = data[i][result[..., i], ...]

        # regularize and stitch
        for i in range(len(data)):
            data[i] = util.regularize(data[i])
        data_processed = util.flatten(data)
        log.log_info('Other processing completed')

    train_x, train_y, test_x, test_y = util.shuffle_data(data_processed, num_good, num_bad, 0.3)

    # log the cost and accuracy
    cost_log = []
    test_log = []
    train_log = []
    x = []

    layer_dims.insert(0, train_x.shape[0])
    dnn = Dnn(layer_dims, mini_batch_size=train_x.shape[1], learning_rate=learning_rate, lambd=lambd,
              keep_prob=keep_prob)
    dnn.initialize_parameters()
    dnn.forward()
    cost = dnn.cost()
    optimizer = dnn.get_optimizer(cost)
    predict, accuracy = dnn.predict()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        if enable_debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        for i in range(num_iterations):
            train_accuracy = sess.run(accuracy, feed_dict={dnn.x: train_x, dnn.y: train_y})
            # a = sess.run(predict, feed_dict={dnn.x: train_x, dnn.y: train_y})
            test_accuracy = sess.run(accuracy, feed_dict={dnn.x: test_x, dnn.y: test_y})
            cost_value = sess.run(cost, feed_dict={dnn.x: train_x, dnn.y: train_y})
            sess.run(optimizer, feed_dict={dnn.x: train_x, dnn.y: train_y})

            cost_log.append(cost_value)
            train_log.append(train_accuracy)
            test_log.append(test_accuracy)
            x.append(i)

            if print_detail and i % 10 == 0:
                info = '\nIteration %d\n' % i + 'Cost: %f\n' % cost_value + 'Train accuracy: %f\n' % train_accuracy + 'Test accuracy: %f' % test_accuracy
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

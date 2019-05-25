from argparse import ArgumentParser

import tensorflow as tf
# for debug
from tensorflow.python import debug as tf_debug
import numpy as np
from datetime import datetime

from log import log
import util
from config import *
from cnn import Cnn


def train():
    TIMESTAMP = "{0:%Y-%m-%d-%H-%M/}".format(datetime.now())
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
    train_x = np.vstack((train_good_data, train_bad_data_origin))
    train_y = np.array([1.] * len(train_good_data) + [0.] * len(train_bad_data_origin), dtype=np.float).reshape(
        (len(train_bad_data_origin) + len(train_good_data), 1))

    train_x, train_y, num_expand = util.expand(train_x, train_y)

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
    log.log_info('full connected layers: ' + str(fc_layers))
    log.log_info('learning rate: %f' % learning_rate)
    log.log_info('keep prob: ' + str(keep_prob))
    log.log_info('the number of expanding bad data: ' + str(num_expand))
    log.log_info('mini batch size: ' + str(mini_batch_size))

    if mini_batch_size != 0:
        assert mini_batch_size <= len(train_x)

    cnn = Cnn(conv_layers, fc_layers, filters, learning_rate)
    (m, n_W0, n_C0) = train_x.shape
    n_y = train_y.shape[1]

    # construction calculation graph
    cnn.initialize(n_W0, n_C0, n_y)
    cost = cnn.cost()
    optimizer = cnn.get_optimizer(cost)
    predict, accuracy = cnn.predict()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # log for tensorboard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("log/tsb/train/" + TIMESTAMP, sess.graph)
        test_writer = tf.summary.FileWriter("log/tsb/test/" + TIMESTAMP)

        if enable_debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        sess.run(init)

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
            summary_train, train_accuracy = sess.run([merged, accuracy],
                                                     feed_dict={cnn.x: train_x, cnn.y: train_y,
                                                                cnn.keep_prob: 1})
            summary_test, test_accuracy = sess.run([merged, accuracy],
                                                   feed_dict={cnn.x: test_x, cnn.y: test_y, cnn.keep_prob: 1})

            train_writer.add_summary(summary_train, i - 1)
            test_writer.add_summary(summary_test, i - 1)

            if print_detail and (i % 10 == 0 or i == 1):
                info = '\nIteration %d\n' % i + \
                       'Cost: %f\n' % cost_value + \
                       'Train accuracy: %f\n' % train_accuracy + \
                       'Test accuracy: %f' % test_accuracy
                log.log_info(info)

            # stop when test>0.95 and train>0.99
            if test_accuracy >= 0.95 and train_accuracy >= 0.99:
                info = '\nIteration %d\n' % i + \
                       'Cost: %f\n' % cost_value + \
                       'Train accuracy: %f\n' % train_accuracy + \
                       'Test accuracy: %f' % test_accuracy
                log.log_info(info)
                saver.save(sess, "log/model/" + TIMESTAMP)
                break
            saver.save(sess, "log/model/" + TIMESTAMP)
        train_writer.close()
        test_writer.close()

    log.log_info('program end')


def predict(path: str, data_x: np.ndarray):
    # Pretreatment
    data_x, length = util.resample(data_x, 600)
    data_x = util.reshape(data_x, length)
    for i in range(len(data_x)):
        data_x[i, :, 0] = util.regularize(data_x[i, :, 0])
        data_x[i, :, 1] = util.regularize(data_x[i, :, 1])
        data_x[i, :, 2] = util.regularize(data_x[i, :, 2])

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(path + '.meta')
        saver.restore(sess, path)
        graph = tf.get_default_graph()
        placehold_x = graph.get_tensor_by_name('input/data_x:0')
        predict_value = graph.get_tensor_by_name('accuracy/predict:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        return sess.run(predict_value, feed_dict={placehold_x: data_x, keep_prob: 1})


if __name__ == '__main__':
    parser = ArgumentParser(description='Welding evaluation system')
    subparsers = parser.add_subparsers(dest='command', description='command')

    grade_parser = subparsers.add_parser('train', help='train the model. The parameter is defined in config.py')

    class_parser = subparsers.add_parser('predict', help='predict the data with prepared weights')
    class_parser.add_argument('start index', type=int, help='the start index of data')
    class_parser.add_argument('end index', type=int, help='the end index of data')
    class_parser.add_argument('weight', type=str, help='the path of the weight file. You should end with file separator')

    args = vars(parser.parse_args())
    if args['command'] == 'train':
        train()
    elif args['command'] == 'predict':
        print(predict(args['weight'], util.load_data_example(args['start index'], args['end index'])))

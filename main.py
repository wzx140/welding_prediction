import matplotlib.pyplot as plt
import tensorflow as tf
# for debug
from tensorflow.python import debug as tf_debug

import log
import util
from config import *
from cnn import Cnn

if __name__ == '__main__':
    log.log_debug('Data loading begin')
    data, num_good, num_bad = util.load_data(50)
    log.log_info('Data loading completed')

    # resample and regularize
    data, length = util.resample(data, 512)
    data = util.reshape(data, length)
    for i in range(len(data)):
        data[i, :, 0] = util.regularize(data[i, :, 0])
        data[i, :, 1] = util.regularize(data[i, :, 1])
        data[i, :, 2] = util.regularize(data[i, :, 2])
    train_x, train_y, test_x, test_y = util.shuffle_data(data, num_good, num_bad, 0.3)
    log.log_info('prepare completed')

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
            # num_mini_batches = int(m / mini_batch_size)
            # mini_batches = util.random_mini_batches(train_x, train_y, mini_batch_size)

            # cost_value = 0
            # for mini_batch in mini_batches:
            #     (mini_batch_x, mini_batch_y) = mini_batch
            #     _, temp_cost = sess.run([optimizer, cost], feed_dict={cnn.x: mini_batch_x, cnn.y: mini_batch_y})
            #     cost_value += temp_cost
            # cost_value /= num_mini_batches

            _, cost_value = sess.run([optimizer, cost], feed_dict={cnn.x: train_x, cnn.y: train_y})

            train_accuracy, output = sess.run([accuracy, predict], feed_dict={cnn.x: train_x, cnn.y: train_y})
            test_accuracy = sess.run(accuracy, feed_dict={cnn.x: test_x, cnn.y: test_y})

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

import tensorflow as tf
import numpy as np


class Cnn(object):

    def __init__(self, conv_layers, fc_layers, filters, learning_rate):
        self.__param = {}
        self.__conv_layers = conv_layers
        self.__fc_layers = fc_layers
        self.__filters = filters
        self.__lr = learning_rate
        self.__keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def keep_prob(self):
        return self.__keep_prob

    def __variable_summaries(self, var):
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)

    def initialize(self, n_w0, n_c0, n_y):
        """
        initialize the w, X, Y
        :param n_w0: width of the image data
        :param n_c0: depth of the image data
        :param n_y: number of the labels
        :return:
        """
        f = self.__filters
        convs = self.__conv_layers
        with tf.name_scope('input'):
            self.__x = tf.placeholder(tf.float32, (None, n_w0, n_c0), 'data_x')
            self.__y = tf.placeholder(tf.float32, (None, n_y), 'data_y')

        # first convolution layer
        with tf.name_scope('conv1'):
            with tf.name_scope('weights'):
                self.__param['W1'] = tf.get_variable('W1', (f[0][0], n_c0, convs[0]),
                                                     initializer=tf.contrib.layers.xavier_initializer())
                self.__variable_summaries(self.__param['W1'])

        conv_index = 1
        for i in range(1, len(convs)):
            if convs[i] != 0 and convs[i] != -1:
                with tf.name_scope('conv' + str(conv_index + 1)):
                    self.__param['W' + str(conv_index + 1)] = tf.get_variable('W' + str(conv_index + 1), (
                        f[i][0], self.__param['W' + str(conv_index)].shape[2], convs[i]),
                                                                              initializer=tf.contrib.layers.xavier_initializer())
                conv_index += 1

    def forward(self):
        param = self.__param
        f = self.__filters
        convs = self.__conv_layers
        fcs = self.__fc_layers
        a_pre = self.__x

        conv_index = 0

        # convolution forward
        for i in range(len(convs)):
            with tf.name_scope('conv' + str(conv_index + 1)):
                if convs[i] == 0:
                    with tf.name_scope('max_pool'):
                        a = tf.nn.pool(a_pre, window_shape=[f[i][0]], padding=f[i][2], pooling_type="MAX")
                    a_pre = a
                elif convs[i] == -1:
                    with tf.name_scope('dropout'):
                        a = tf.nn.dropout(a_pre, keep_prob=self.__keep_prob)
                        tf.summary.scalar('dropout_keep_probability', self.__keep_prob)
                    a_pre = a
                else:
                    with tf.name_scope('output'):
                        z = tf.nn.conv1d(a_pre, param['W' + str(conv_index + 1)], stride=f[i][1], padding=f[i][2])
                        a = tf.nn.relu(z)
                    a_pre = a
                    conv_index += 1

        # full connected forward
        a_pre = tf.contrib.layers.flatten(a)
        a_pre = tf.nn.dropout(a_pre, keep_prob=self.__keep_prob)
        for i in range(len(fcs)):
            a = tf.contrib.layers.fully_connected(a_pre, fcs[i])
            a_pre = a
        a = tf.contrib.layers.fully_connected(a_pre, 1, activation_fn=None)
        self.__a = a

    def cost(self):
        with tf.name_scope("loss"):
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.__a, labels=self.__y))
        tf.summary.scalar("loss", cost)
        return cost

    def get_optimizer(self, cost):
        with tf.name_scope('train'):
            adam = tf.train.AdamOptimizer(self.__lr).minimize(cost)
        return adam

    def predict(self):
        pre = tf.cast(tf.greater(self.__a, 0.5), dtype=np.float, name='output')
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(pre, self.__y), "float"))
            tf.summary.scalar("accuracy", accuracy)

        # calculate f1_score
        TP = tf.count_nonzero(pre * self.__y)
        TN = tf.count_nonzero((pre - 1) * (self.__y - 1))
        FP = tf.count_nonzero(pre * (self.__y - 1))
        FN = tf.count_nonzero((pre - 1) * self.__y)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return pre, accuracy, f1

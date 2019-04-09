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
        self.__x = tf.placeholder(tf.float32, (None, n_w0, n_c0), 'data_x')
        self.__y = tf.placeholder(tf.float32, (None, n_y), 'data_y')

        # first convolution layer
        self.__param['W1'] = tf.get_variable('W1', (f[0][0], n_c0, convs[0]),
                                             initializer=tf.contrib.layers.xavier_initializer())
        conv_index = 1
        for i in range(1, len(convs)):
            if convs[i] != 0 and convs[i] != -1:
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
            if convs[i] == 0:
                a = tf.nn.pool(a_pre, window_shape=[f[i][0]], padding=f[i][2], pooling_type="MAX")
                a_pre = a
            elif convs[i] == -1:
                a = tf.nn.dropout(a_pre, keep_prob=self.__keep_prob)
                a_pre = a
            else:
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
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.__a, labels=self.__y))
        return cost

    def get_optimizer(self, cost):
        adam = tf.train.AdamOptimizer(self.__lr).minimize(cost)
        return adam

    def predict(self):
        pre = tf.cast(tf.greater(self.__a, 0.5), dtype=np.float, name='output')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pre, self.__y), "float"))

        # calculate f1_score
        TP = tf.count_nonzero(pre * self.__y)
        TN = tf.count_nonzero((pre - 1) * (self.__y - 1))
        FP = tf.count_nonzero(pre * (self.__y - 1))
        FN = tf.count_nonzero((pre - 1) * self.__y)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return pre, accuracy, f1

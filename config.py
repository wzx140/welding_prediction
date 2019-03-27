# numbers of neurons in each convolution layer, 0 is the max pool, -1 is the dropout
conv_layers = [18, 0, 36, 0, 72, -1, 0, 144]

# (filter size, step, pad) in filters in each layers. For pad, fill in 'SAME' or 'VALID'.
# Must correspond to conv_layers. If it is dropout, fill with 0
filters = [(2, 1, 'SAME'), (2, 2, 'SAME'), (2, 1, 'SAME'), (2, 2, 'SAME'), (2, 1, 'SAME'), 0, (2, 2, 'SAME'),
           (2, 1, 'SAME')]

# the dims of full connected layers. The last layer is 1 and you do not need to write it on
fc_layers = []

learning_rate = 0.00005

num_epochs = 100

# 0->disable
mini_batch_size = 64

# display and record some detail
print_detail = True

enable_debug = False

# regularization
keep_prob = 0.3

# expand train data
add_noise_radio = 0.5
# the number of data for train and test
num_data = 1000

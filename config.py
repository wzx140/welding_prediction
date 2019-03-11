# numbers of neurons in each convolution layer, 0 is the max pool, -1 is the average pool
conv_layers = [8, 0, 16, 0]

# (filter size, step, pad) in filters in each layers. For pad, fill in 'SAME' or 'VALID'.
filters = [(4, 1, 'SAME'), (8, 8, 'SAME'), (2, 1, 'SAME'), (4, 4, 'SAME')]

# the dims of full connected layers. The last layer is 1 and you do not need to write it on
fc_layers = [64]

learning_rate = 0.001

num_epochs = 200

# NOT USE
mini_batch_size = 64

# display and record some detail
print_detail = True

enable_debug = False

# regularization NOT USE
lambd = 0
keep_prob = 1

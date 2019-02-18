# number of nodes in each layer exclude input layer
layer_dims = [2000, 7, 4, 1]

learning_rate = 0.000075

num_iterations = 500

# display and record some detail
print_detail = True

# fast mode: read preprocessed data directly
fast_mode = True

# incomplete mode: randomly select train data in order to ensure the sizes of train data and test data are equal
incomplete_mode = True

enable_debug = False

# regularization
lambd = 0
keep_prob = 0.6

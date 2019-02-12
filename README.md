## Deep Neural Network For Predict Welding Quality
### PREREQUISITES
- numpy
- h5py
- matplotlib
- tensorflow-gpu<sup>*</sup>
> Tensorflow without GPU is also feasible but running slowly

### Features
#### DTW
Use *dynamic time warping* to pre-process data

#### Regularization
Implemented Dropout and L2-regularization
- set *keep_prob* in *config.py* range from 0~1. 1 means Dropout is disabled
- set *lambd* in *config.py* range from 0~1. 0 means L2-regularization is disabled
> **You can not enable both at same time**

#### FAST MOOD
Since it takes a lot of time to complete DTW, you can set *fast_mode* True in *config.py*. It loads the data after DTW

#### PERSISTENCE
The log is in *log.txt* when program execution completed
 
### RUN
Other variables that can be changed are in *config.py*. Run 'python main.py' in project dir

### DEBUG
If you want to use **tfdbg**, you should,
- install *pyreadline* by pip
- set *enable_debug* True in *config.py*
- run `python main.py --debug` in project dir
> For more information, you can read [official document](https://www.tensorflow.org/guide/debugger)

### MORE
There are some problems
- Uneven distribution of training data
- Temporarily did not get a good network structure by experiment
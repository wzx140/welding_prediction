## Convolutional Neural Network For Predicting Welding Quality
### PREREQUISITES
- numpy
- h5py
- matplotlib
- git LFS
- tensorflow-gpu<sup>*</sup>
> Tensorflow without GPU is also feasible but running slowly

### Introduction
According to the curve waveform of voltage, current and electrode position, judge the quality of flash welding. The data is multi-dimensional time series. We have 2000 of good quality and 50 of bad quality. In this network, I just load 50 of good quality avoiding data imbalance. **Experiments show that CNN is more effective than BP-network.** I think convolution can identify the relative positional relationship between multi-dimensional time series, which reduces the over-fitting of the model to some extent.

### Features

#### Regularization
~~Implemented Dropout and L2-regularization~~
- ~~set *keep_prob* in *config.py* range from 0~1. 1 means Dropout is disabled~~
- ~~set *lambd* in *config.py* range from 0~1. 0 means L2-regularization is disabled~~
> ~~**You can not enable both at same time**~~

Next time will add this feature

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
- use `git LFS` to download the data

There are some problems
- Uneven distribution of training data
- Temporarily did not get a good network structure by experiment
## Convolutional Neural Network For Predicting Welding Quality
### PREREQUISITES
- numpy
- h5py
- matplotlib
- git LFS
- tensorflow-gpu<sup>*</sup>
> Tensorflow without GPU is also feasible but running slowly

### Introduction
According to the curve waveform of voltage, current and electrode position, judge the quality of flash welding. The data is multi-dimensional time series. We have 2000 of good quality and 50 of bad quality. In this network, I use data augmentation to increase the number of bad. **Experiments show that CNN is more effective than BP-network and Dropout is effective.** I think convolution can identify the relative positional relationship between multi-dimensional time series, which reduces the over-fitting of the model to some extent.

### Features

#### Regularization
Implemented Dropout
- set *keep_prob* in *config.py* range from 0~1. 1 means Dropout is disabled

#### PERSISTENCE
The log is in *log.txt* when program execution completed
 
####  DATA AUGMENTATION
Since we only have 50 bad samples, we use transition and adding noise to expand the bad samples. You can change the number of data we used to train in `config.py`. But we always keep number of bad equals good by equip noise and transtition.
- Adding noise means that add random number to each time of each dim of data. In my project, I use -0.5~0.5.
    - you can change the ratio of add noise number in `config.py`
- Transtition means that make the data whole up or down. In my project, I use -1~1
    - the rest of sample we need except adding noise is generate by transtition
 
### RUN
Other variables that can be changed are in *config.py*. Run `python main.py` in project dir

### DEBUG
If you want to use **tfdbg**, you should,
- install *pyreadline* by pip
- set *enable_debug* True in *config.py*
- run `python main.py --debug` in project dir
> For more information, you can read [official document](https://www.tensorflow.org/guide/debugger)

### MORE
- use `git LFS` to download the data
- model's accuracy is about 0.95
- see the [demo](./demo.ipynb)
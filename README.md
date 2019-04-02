## Convolutional Neural Network For Predicting Welding Quality
### PREREQUISITES
- numpy
- h5py
- matplotlib
- git LFS
- tensorflow

### OPTIONAL
- tensorflow-gpu
- prettytable

### Introduction
According to the curve waveform of voltage, current and electrode position, judge the quality of flash welding. The data is multi-dimensional time series. We have 2000 of good quality and 50 of bad quality. In this network, I use data augmentation to increase the number of bad. **Experiments show that CNN is more effective than BP-network and Dropout is effective.** I think convolution can identify the relative positional relationship between multi-dimensional time series, which reduces the over-fitting of the model to some extent.

### Features

#### Regularization
Implemented Dropout
- set *keep_prob* in *config.py* range from 0~1. 1 means Dropout is disabled

####  DATA AUGMENTATION
Since we only have 50 bad samples, we use transition and adding noise to expand the bad samples. You can change the number of data we used to train in `config.py`. But we always keep number of bad equals good by equip noise and transtition.
- Adding noise means that add random number to each time of each dim of data. In my project, I use -0.5~0.5.
    - you can change the ratio of add noise number in `config.py`
- Transtition means that make the data whole up or down. In my project, I use -1~1
    - the rest of sample we need except adding noise is generate by transtition
 
#### VISUALIZATION
The log is in `log/log.txt` when program execution completed. You can visualize the data in `log/log.txt` with a nice table. You should install prettytable with pip

Before you run `visualization.py`, you should have more than one complete operational process with `main.py`

### RUN
- `git clone git@github.com:wzx140/welding_prediction.git`. This process may be very slow for we also downloading the data for train by `git LFS` 
- Other variables that can be changed are in *config.py*. 
- `cd welding_prediction`
- `python main.py`

For visualization
- `python log/visualization.py`

### DEBUG
If you want to use **tfdbg**, you should,
- install *pyreadline* by pip
- set *enable_debug* True in *config.py*
- run `python main.py --debug` in project dir
> For more information, you can read [official document](https://www.tensorflow.org/guide/debugger)

### MORE
- model's accuracy is about 0.95
- see the [demo](./demo.ipynb)
## 基于卷积神经网络的焊接评估系统
### 必备库
- numpy
- tensorflow
- imblearn

### 可选择的
- tensorflow-gpu
- sklearn
- matplotlib

### 引言
根据电压，电流和电极位置的曲线波形，判断闪光焊接的质量。数据是多维时间序列。我们有2000个质量好，50个质量差储存在`./data/`中。 在这个网络中，我使用数据扩充来增加坏的数量。 **实验表明，CNN比BP网络更有效，Dropout是有效的。**我认为卷积可以识别多维时间序列之间的相对位置关系，这在一定程度上减少了模型的过度拟合。 如下所示，原始数据是多维时间序列。
<img src="img/data.png" width = "50%" />

### 开始
- `git clone git@github.com:wzx140/welding_prediction.git`
- 修改 *config.py* 中的参数
- `cd welding_prediction`
- `python main.py train`，训练模型并将模型保存在`log/mode`中。此文件夹中有训练好的模型
- `tensorboard --logdir log/tsb`, 可以看到训练过程参数的可视化结果
- `python main.py predict path-to-mode path-to-sample` 去预测焊接质量

### 特点

#### 正则化
应用了 Dropout
- 设置 *keep_prob* 在 *config.py* 的值在 0~1 之间. 1说明Dropout不起作用

####  数据放大
由于我们只有50的质量为坏的样本，所以使用ADASYN去扩大负样本。更多的信息可以阅读我的博客 [ADASYN](https://masterwangzx.com/2019/04/08/SMOTE/#adasyn)

#### TENSORBOARD
运行 `main.py`之后, tensorboard的数据储存在 `log/tsb`. 只需运行 `tensorboard --logdir log/tsb`

#### DEBUG
如果你想要使用 **tfdbg**, 你应该,
- 使用pip安装 *pyreadline* 
- 令 *enable_debug* 参数的值为True 在 *config.py* 里
- 运行 `python main.py train --debug` 在工程文件夹里
> 更多的信息, 你可以阅读 [官方文档](https://www.tensorflow.org/guide/debugger)

## 示例
- 看 [示例](./demo.ipynb)
- 我们实现了 [DTW](./other/DTW.ipynb) 和 [KNN](./other/KNN.ipynb) 去分类我们的模型.

## 更多
- 卷积神经网络模型的训练集精度大于0.99，测试集精度大于0.96
- 最好的网络结构是 (18 - 36 - 72 - 144)
![](img/net.png)

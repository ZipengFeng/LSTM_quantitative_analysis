import pandas as pd
import numpy as np

from random import shuffle
import os
# from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.contrib import learn

import matplotlib.pyplot as plt
%matplotlib inline

# 提取数据
data = []
date = []
files = os.listdir('/home/chocolate/MCdata')
files = sorted(files)
for f_name in files:
    if f_name.startswith('MCMINU_IF'):
        data_tmp = pd.read_csv(
            '/home/chocolate/MCdata/' + f_name, usecols=[0, 2])
        GbData = data_tmp.groupby(['Date'])
        for time, group in GbData:
            date.append(time)
            price_data = group['Latestprice'].values
            beginPrice = price_data[0]
            raiseDownData = map(lambda x: (
                x - beginPrice) / beginPrice, price_data)
            data.append(raiseDownData[10:-30])
a = pd.to_datetime(date)
date = map(lambda y: y.strftime('%Y-%m-%d'), a)

# 分割成训练集与测试集，训练集占80%
test_size = 0.2
data_len = len(data)
nTest = int(data_len * test_size)

train_data = data[0:data_len - nTest]
train_date = date[0:data_len - nTest]
test_data = data[data_len - nTest:]
test_date = date[data_len - nTest:]


# 用前25分中的数据对后面第3分钟的数据进行预测
def rnn_data(data, date, timeStep=25, output=3, get_label=False):
    rnn_df = []
    rnn_time = []
    for t in range(len(data)):
        data_tmp = data[t]
        if not get_label:
            for i in range(len(data_tmp) - timeStep - output + 1):
                rnn_df.append(data_tmp[i:i + timeStep])
                rnn_time.append(date[t])
        else:
            for i in range(timeStep, len(data_tmp) - output + 1):
                rnn_df.append(data_tmp[i + output - 1:i + output])
                rnn_time.append(date[t])
    return np.array(rnn_df), rnn_time


train_x, train_x_date = rnn_data(train_data, train_date)
train_y, train_y_date = rnn_data(train_data, train_date, get_label=True)
zip_train = zip(train_x, train_y)
shuffle(zip_train)
train_x, train_y = zip(*zip_train)
train_x = np.array(train_x)
train_y = np.array(train_y)

test_x, test_x_date = rnn_data(test_data, test_date)
test_y, test_y_date = rnn_data(test_data, test_date, get_label=True)


# 将时间窗口序列分割为时间点数据
def input_fn(X):
    return tf.split(1, 25, X)


# 设置rnn回归模型参数
TRAINING_STEPS = 8000
BATCH_SIZE = 2000
regressor = learn.TensorFlowRNNRegressor(rnn_size=25, cell_type='lstm', steps=TRAINING_STEPS,
                                         learning_rate=0.003, batch_size=BATCH_SIZE, verbose=2, input_op_fn=input_fn)
regressor.fit(train_x, train_y)


# 画图，画出预测值与真实值的对比图
predicted = regressor.predict(test_x)
plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
ax.plot(test_y[200:400], label='test')
ax.plot(predicted[200:400], label='predicted')
plt.legend()

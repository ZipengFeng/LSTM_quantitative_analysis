import pandas as pd
import numpy as np
import os
# from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.ops import rnn, rnn_cell
import matplotlib.pyplot as plt
%matplotlib inline

# 提取数据与特征选择
data = []
date = []
files = os.listdir('/home/chocolate/MCdata')
files = sorted(files)
for f_name in files:
    if f_name.startswith('MCMINU_IF16'):
        data_tmp = pd.read_csv(
            '/home/chocolate/MCdata/' + f_name,
            usecols=[0, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
        GbData = data_tmp.groupby(['Date'])
        for time, group in GbData:
            date.append(time)
            # 读取表格数据
            firstPrice = group['First.Latestprice'].values  # 每分钟开盘价
            latestPrice = group['Latestprice'].values  # 每分钟收盘价
            maxPrice = group['MaxPrice'].values  # 每分钟最大值
            minPrice = group['MinPrice'].values  # 每分钟最小值
            stockup = group['Stockup'].values    # 每分钟增仓量
            volume = group['Volume'].values      # 每分钟成交量
            buyPrice = group['Last.Buy1price'].values    # 每分钟买一价
            buyQuantity = group['Last.Buy1quantity'].values  # 每分钟买一量
            sellPrice = group['Last.Sell1price'].values  # 每分钟卖一价
            sellQuantity = group['Last.Sell1quantity'].values  # 每分钟卖一量

            latestRaiseDown = map(lambda x: (
                x - firstPrice) / firstPrice, latestPrice)  # 收盘涨跌幅

            maxRaiseDown = map(lambda x: (
                x - firstPrice) / firstPrice, maxPrice)    # 最大值涨跌幅
            minRaiseDown = map(lambda x: (
                x - firstPrice) / firstPrice, minPrice)    # 最小值涨跌幅
# 特征未写完，正在补充中

            data.append()
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
train_x = np.array(train_x)
train_y = np.array(train_y)

test_x, test_x_date = rnn_data(test_data, test_date)
test_y, test_y_date = rnn_data(test_data, test_date, get_label=True)


# 构建五类的分类器，用涨跌幅度作为阈值，分别为：大涨、小涨、平稳、小跌、大跌
train_RaiseDown = []
for i in train_y:
    if i > 0.002:
        train_RaiseDown.append([1, 0, 0, 0, 0])
    elif i > 0.0005:
        train_RaiseDown.append([0, 1, 0, 0, 0])
    elif i > -0.0005:
        train_RaiseDown.append([0, 0, 1, 0, 0])
    elif i > -0.002:
        train_RaiseDown.append([0, 0, 0, 1, 0])
    else:
        train_RaiseDown.append([0, 0, 0, 0, 1])
train_y = np.array(train_RaiseDown)

test_RaiseDown = []
for i in test_y:
    if i > 0.002:
        test_RaiseDown.append([1, 0, 0, 0, 0])
    elif i > 0.0005:
        test_RaiseDown.append([0, 1, 0, 0, 0])
    elif i > -0.0005:
        test_RaiseDown.append([0, 0, 1, 0, 0])
    elif i > -0.002:
        test_RaiseDown.append([0, 0, 0, 1, 0])
    else:
        test_RaiseDown.append([0, 0, 0, 0, 1])
test_y = np.array(test_RaiseDown)


# # 将时间窗口序列分割为时间点数据
# def input_fn(X):
#     return tf.split(1, 25, X)


# # 设置rnn回归模型参数
# TRAINING_STEPS = 8000
# BATCH_SIZE = 2000
# classifier = learn.TensorFlowRNNClassifier(rnn_size = 50,n_classes=1,cell_type = 'lstm', steps=TRAINING_STEPS,
#                                         learning_rate=0.003, batch_size=BATCH_SIZE, verbose=2, input_op_fn=input_fn)
# classifier.fit(train_x, train_y, steps = 25, monitors=None)


# 参照LSTM网络训练mnist测试集的程序进行修改，来对我们的模型进行分类
#  动态设置步长
def next_batch(step, batch_size):
    return train_x[(step - 1) * batch_size:step * batch_size],
    train_y[(step - 1) * batch_size:step * batch_size]


# 神经网络参数设置
learning_rate = 0.001
training_iters = 100000
batch_size = 150
display_step = 10
n_input = 3
n_steps = 50
n_hidden = 150
n_classes = 5


x = tf.placeholder("float", [None, n_steps, n_input])
istate = tf.placeholder("float", [None, 2 * n_hidden])
y = tf.placeholder("float", [None, n_classes])


weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(_X, _istate, _weights, _biases):

    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size

    _X = tf.reshape(_X, [-1, n_input])  # (n_steps*batch_size, n_input)

    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    _X = tf.split(0, n_steps, _X)  # n_steps * (batch_size, n_hidden)

    outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']


pred = RNN(x, istate, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# 评估模型
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化变量
init = tf.initialize_all_variables()

# 建立会话
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # 保持训练直到最大迭代
    while step * batch_size < training_iters:
        batch_xs, batch_ys = next_batch(step, batch_size)
        # Reshape data to get 28 seq of 28 elements
        if len(batch_xs) != batch_size:
            break
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2 * n_hidden))})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs,
                                                y: batch_ys,
                                                istate: np.zeros((batch_size, 2 * n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate: np.zeros((batch_size, 2 * n_hidden))})
            print "Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"
    # Calculate accuracy for 256 mnist test images
    test_data = test_x.reshape((-1, n_steps, n_input))
    test_label = test_y
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                             istate: np.zeros((len(test_data), 2 * n_hidden))})

# 画图，画出预测值与真实值的对比图
predicted = classifier.predict(test_x)
plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
ax.plot(test_y[200:400], label='test')
ax.plot(predicted[200:400], label='predicted')
plt.legend()

# encoding: UTF-8

"""
基于LSTM预测的交易策略实现
"""
import pandas as pd
from pandas import Series, DataFrame
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import csv
import psutil
import os
from ctaBase import *
from ctaTemplate import CtaTemplate
import tarfile


########################################################################
class ctaLSTM_V4_2(CtaTemplate):
    className = 'ctaLSTM_V4_2'
    author = u'Feng Zipeng'

    # 策略参数
    # buy_raise_score = 4.7  # 预测得分超过该值，建多仓
    # sell_down_score = 3.3
    # buy_down_score = 3.2
    # sell_raise_score = 4.6
    # zhisun = 0.0015  # 止损阈值
    # zhiying = 0.0012  # 止盈阈值
    # seq_len = 25  # 用于预测的时间序列长度
    n_input = 10  # 特征数量
    n_steps = 15  # 时间序列长度
    n_hidden = 500  # 隐藏层神经元个数
    n_classes = 7  # 分类数量

    # 策略变量
    count = 0  # 用来记录接收bar数据的个数
    count_bar = 0  # 用来记录bar推送给onbar数据的个数
    bar = None
    barMinute = EMPTY_STRING
    date = None
    # score = 0
    predict_1 = 0   # 预测每分钟分类
    predict_2 = 0
    predict_3 = 0
    predict_4 = 0
    predict_5 = 0
    real_raisedown = 0   # 当前分钟真实涨跌幅
    real_class = 0    # 当前分钟真实分类
    acc_1 = 0    # 准确率
    acc_2 = 0
    acc_3 = 0
    acc_4 = 0
    acc_5 = 0
    raise_count = 0
    down_count = 0

    rec_price = 0   # 参考价
    init_price = 0
    pos_rec = 0    # 参考持仓量
    position = 0
    last_minutevolume = 0  # 保存上一分钟最后一个TICK的成交量
    latest_minutevolume = 0  # 用来保存当前这一分钟最后一个tick的成交量
    DATAS = []  # 保存bar数据
    # trade_records = []  # 保存交易数据
    # datas = []   # 保存实时价格
    predict_1min = []
    predict_2min = []
    predict_3min = []
    predict_4min = []
    predict_5min = []
    real_1min = []
    # predict_records = []    # 保存预测分类
    # real_records = []     # 保存实际分类
    zhisun_label = False
    zhisun_bar = 0
    count_correct_1 = 0
    count_correct_2 = 0
    count_correct_3 = 0
    count_correct_4 = 0
    count_correct_5 = 0

    # 参数列表，保存了参数的名称
    paramList = ['name',
                 'className',
                 'author',
                 'vtSymbol']

    # 变量列表，保存了变量的名称
    varList = ['inited',
               'trading',
               'pos',
               'pos_rec',
               'count',
               'count_bar',
               'real_class',
               'predict_1',
               'predict_2',
               'predict_3',
               'predict_4',
               'predict_5',
               'acc_1',
               'acc_2',
               'acc_3',
               'acc_4',
               'acc_5']

    # ----------------------------------------------------------------------
    def __init__(self, ctaEngine, setting, filepath='/home/chocolate/LSTM-source/models/model_5.1/models.tar.gz'):
        """Constructor"""
        super(ctaLSTM_V4_2, self).__init__(ctaEngine, setting)
        self.DATAS = []
        # self.predict_records = []
        # self.real_records = []
        # self.trade_records = []
        # self.datas = []
        self.lastOrder = None
        self.filepath = filepath

        # # Open tarfile
        # tar = tarfile.open(mode="r:gz", fileobj=file(self.filepath))

        # for member in tar.getnames():
        #     print tar.extractfile(member).read()

        tar = tarfile.open(self.filepath)   # 解压依赖文件，读取模型
        tar.extractall()
        tar.close()
        # self.writeCtaLog(u'内存占用情况:' + str(self.vi_mem_record()))

        # 注意策略类中的可变对象属性（通常是list和dict等），在策略初始化时需要重新创建，
        # 否则会出现多个策略实例之间数据共享的情况，有可能导致潜在的策略逻辑错误风险，
        # 策略类中的这些可变对象属性可以选择不写，全都放在__init__下面，写主要是为了阅读
        # 策略时方便（更多是个编程习惯的选择）

    # ----------------------------------------------------------------------
    def onInit(self):
        """初始化策略（必须由用户继承实现）"""
        self.writeCtaLog(u'LSTM策略初始化')

        # initData = self.loadBar(self.initDays)
        # for bar in initData:
        #     self.onBar(bar)

        self.putEvent()

    # ----------------------------------------------------------------------
    def onStart(self):
        """启动策略（必须由用户继承实现）"""
        self.writeCtaLog(u'LSTM策略启动')
        self.putEvent()

    # ----------------------------------------------------------------------
    def onStop(self):
        """停止策略（必须由用户继承实现）"""
        self.writeCtaLog(u'LSTM策略停止')

        self.putEvent()

    # ----------------------------------------------------------------------
    def onTick(self, tick):
        """收到行情TICK推送（必须由用户继承实现）"""
        tickMinute = tick.datetime.minute
        self.price = tick.lastPrice   # 记录每个TICK的最新价
        self.bidprice1 = tick.bidPrice1
        self.askprice1 = tick.askPrice1

        if tickMinute != self.barMinute:
            if self.bar:
                self.bar.mean = (self.bar.low + self.bar.close /
                                 + self.bar.high) / 3.0

                self.bar.Stockup = self.bar.openInterest - self.position
                self.position = self.bar.openInterest
                # 保存上一分钟的最后一个TICK的持仓量，方便插入下一分钟数据时计算

                self.bar.volume = self.latest_minutevolume \
                    - self.last_minutevolume
                self.last_minutevolume = self.latest_minutevolume

                # 将上一分钟的数据推送给onBar
                self.count_bar += 1  # 推送给onbar的次数+1
                #     # 在每次tick.minute更新时将上一分钟的bar数据推送给onbar
                self.onBar(self.bar)

            bar = CtaBarData()

            bar.open = tick.lastPrice
            bar.high = tick.lastPrice
            bar.low = tick.lastPrice
            bar.close = tick.lastPrice
            bar.askPrice1 = tick.askPrice1
            bar.bidPrice1 = tick.bidPrice1
            bar.askvo1 = tick.askVolume1
            bar.bidvo1 = tick.bidVolume1

            bar.date = tick.date
            bar.time = tick.time
            bar.datetime = tick.datetime    # K线的时间设为第一个Tick的时间

            bar.openInterest = tick.openInterest  # 持仓量，是每一个tick的开始持仓量
            self.latest_minutevolume = tick.volume
            self.date = bar.date

            self.bar = bar                  # 这种写法为了减少一层访问，加快速度
            self.barMinute = tickMinute     # 更新当前的分钟

        else:                               # 否则继续累加新的K线
            bar = self.bar                  # 写法同样为了加快速度

            bar.high = max(bar.high, tick.lastPrice)
            bar.low = min(bar.low, tick.lastPrice)
            bar.close = tick.lastPrice
            bar.askPrice1 = tick.askPrice1
            bar.bidPrice1 = tick.bidPrice1
            bar.askvo1 = tick.askVolume1
            bar.bidvo1 = tick.bidVolume1

            # 实时记录当前这一分钟最后一个tick的成交量
            self.latest_minutevolume = tick.volume
            bar.openInterest = tick.openInterest

        self.putEvent()

    # ---------------------------------------------------------------------
    def onBar(self, bar):
        """收到Bar推送（必须由用户继承实现）"""
        # start = time.clock()
        new_data = {'close': bar.close, 'max': bar.high, 'min': bar.low,
                    'mean': bar.mean, 'pos': bar.Stockup, 'vol': bar.volume,
                    'open': bar.open, 'askpr1': bar.askPrice1,
                    'askvo1': bar.askvo1,
                    'bidpr1': bar.bidPrice1, 'bidvo1': bar.bidvo1}
# 对买一价和卖一价保持和tick中同样的名称，方便后续调用进行建仓/平仓

        self.DATAS.append(new_data)
        # self.datas.append([bar.datetime, self.price])
        self.count += 1

        # 接收到bar推送，记录实际涨跌分类
        self.writeCtaLog(u"onbar接收到bar推送数据，时间为:" +
                         str(bar.datetime) + u',' + str(self.count) + u'分钟')
        self.writeCtaLog(u'内存占用情况:' + str(self.vi_mem_record()))

        self.real_raisedown = (self.bar.close - self.bar.open) / self.bar.open
        if self.real_raisedown > 0.0006:
            self.real_class = 7
        elif self.real_raisedown > 0.00035:
            self.real_class = 6
        elif self.real_raisedown > 0.0002:
            self.real_class = 5
        elif self.real_raisedown > -0.0002:
            self.real_class = 4
        elif self.real_raisedown > -0.00035:
            self.real_class = 3
        elif self.real_raisedown > -0.0006:
            self.real_class = 2
        else:
            self.real_class = 1

        # 14:58后停止交易，强制平仓，不留过夜仓
        if bar.datetime.hour == 14 and bar.datetime.minute >= 58:
            # print(u'stop time')
            if self.pos == 0:
                pass
            if self.pos > 0:
                self.sell(self.bidprice1, 1)
                self.zhisun_label = True
                # self.trade_records.append([bar.datetime, self.price, u'sell'])
            if self.pos < 0:
                self.cover(self.askprice1, 1)
                self.zhisun_label = True
                # self.trade_records.append([bar.datetime, self.price, u'cover'])

        # 如果是第一次推送，则不接受该数据，因为第一条数据的增仓量是不对的
        if self.count_bar == 1:
            self.writeCtaLog(u"onbar第一次接受数据，数据不准确，不接收.")

        #  前26分钟的数据只接收，不作判定
        if self.count >= 16:
            datafr = pd.DataFrame(self.DATAS)
            # 获得添加特征并整理后的时间序列数据
            seq_data = self.data_use(datafr)
            # 获得1-5分钟涨跌情况预测类别
            self.predict_1, self.predict_2, self.predict_3,\
                self.predict_4, self.predict_5 = self.pred(seq_data)

            # 记录实际涨跌分类和预测涨跌分类
            self.predict_1min.append(self.predict_1)
            self.predict_2min.append(self.predict_2)
            self.predict_3min.append(self.predict_3)
            self.predict_4min.append(self.predict_4)
            self.predict_5min.append(self.predict_5)
            self.real_1min.append(self.real_class)
            # print predict_records
            # print real_records
            # self.predict_records.append([bar.datetime, self.predict_1, self.predict_2, self.predict_3, self.predict_4, self.predict_5])
            # self.real_records.append([bar.datetime, self.real_class])

            # 计算1-5分钟的预测准确率
            count_f = float(self.count)
            if len(self.predict_1min) >= 2:
                if self.predict_1min[-2:-1] == self.real_1min[-1:]:
                    self.count_correct_1 += 1
            count_correct_f1 = float(self.count_correct_1)
            self.acc_1 = count_correct_f1 / (count_f - 15.0)

            if len(self.predict_2min) >= 3:
                if self.predict_2min[-3:-2] == self.real_1min[-1:]:
                    self.count_correct_2 += 1
            count_correct_f2 = float(self.count_correct_2)
            self.acc_2 = count_correct_f2 / (count_f - 15.0)

            if len(self.predict_3min) >= 4:
                if self.predict_3min[-4:-3] == self.real_1min[-1:]:
                    self.count_correct_3 += 1
            count_correct_f3 = float(self.count_correct_3)
            self.acc_3 = count_correct_f3 / (count_f - 15.0)

            if len(self.predict_4min) >= 5:
                if self.predict_4min[-5:-4] == self.real_1min[-1:]:
                    self.count_correct_4 += 1
            count_correct_f4 = float(self.count_correct_4)
            self.acc_4 = count_correct_f4 / (count_f - 15.0)

            if len(self.predict_5min) >= 6:
                if self.predict_5min[-6:-5] == self.real_1min[-1:]:
                    self.count_correct_5 += 1
            count_correct_f5 = float(self.count_correct_5)
            self.acc_5 = count_correct_f5 / (count_f - 15.0)

            self.writeCtaLog(u"预测准确率分别为：" + str(self.acc_1, self.acc_2, self.acc_3, self.acc_4, self.acc_5))

            # 计算加权得分
            # self.score = 0.3 * self.predict_1 + 0.2 * self.predict_2 + \
            #     0.2 * self.predict_3 + 0.2 * self.predict_4 + \
            #     0.1 * self.predict_5

            self.raise_count = 0
            self.down_count = 0

            if self.predict_1 >= 5:
                self.raise_count += 1
            elif self.predict_1 <= 3:
                self.down_count += 1

            if self.predict_2 >= 5:
                self.raise_count += 1
            elif self.predict_2 <= 3:
                self.down_count += 1

            if self.predict_3 >= 5:
                self.raise_count += 1
            elif self.predict_3 <= 3:
                self.down_count += 1

            if self.predict_4 >= 5:
                self.raise_count += 1
            elif self.predict_4 <= 3:
                self.down_count += 1

            if self.predict_5 >= 5:
                self.raise_count += 1
            elif self.predict_5 <= 3:
                self.down_count += 1

            self.pos_rec_concert()
            # 考虑到发出建仓/平仓信号但是没有成功交易的情况，强制更新pos_rec与pos一致

            if self.pos == 0:
                self.zhisun_set()
                # 当前持仓为0，如果上一笔交易是止损平仓，则考虑停止5分钟再进行建仓判断

                # 得分大于买多阈值时，建多仓
                if self.raise_count >= 2 and self.down_count <= 1 and self.zhisun_label == False:
                    self.buy(self.askprice1, 1)
                    self.rec_price = self.askprice1  # 记录当前的价格作为比较价格
                    self.pos_rec += 1
                    self.writeCtaLog(u'buy!' + str(self.rec_price) + u'pos_rec' + str(self.pos_rec))
                    # self.trade_records.append([bar.datetime, self.price, u'buy'])
                    self.init_price = self.askprice1

                # 得分小于买空阈值时，建空仓
                if self.raise_count <= 1 and self.down_count >= 2 and self.zhisun_label == False:
                    self.short(self.bidprice1, 1)
                    self.rec_price = self.bidprice1
                    self.init_price = self.bidprice1
                    self.pos_rec -= 1
                    self.writeCtaLog(u'short!' + str(self.rec_price) + u'pos_rec' + str(self.pos_rec))
                    # self.trade_records.append([bar.datetime, self.price, u'short'])

            # 平多仓
            if self.pos > 0 and self.pos_rec > 0:
                self.long_pos_sell(bar)

            # 平空仓
            if self.pos < 0 and self.pos_rec < 0:
                self.short_pos_cover(bar)
        # 保存当天数据和交易记录
        # path = "/home/chocolate/LSTM-source/daily_datas/"
        # self.saveFile = file(path + str(self.date) + 'datas.csv', 'wb')
        # self.writer = csv.writer(self.saveFile)
        # self.writer.writerows(self.datas)
        # self.saveFile.close()

        # self.saveFile = file(path + str(self.date) + 'records.csv', 'wb')
        # self.writer = csv.writer(self.saveFile)
        # self.writer.writerows(self.trade_records)
        # self.saveFile.close()

        # self.saveFile = file(path + str(self.date) + 'predict_class.csv', 'wb')
        # self.writer = csv.writer(self.saveFile)
        # self.writer.writerows(self.predict_records)
        # self.saveFile.close()

        # self.saveFile = file(path + str(self.date) + 'real_class.csv', 'wb')
        # self.writer = csv.writer(self.saveFile)
        # self.writer.writerows(self.real_records)
        # self.saveFile.close()

        # 发出状态更新事件
        self.putEvent()

    def RNN(self, x, weights, biases):

        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.n_input])
        x = tf.split(0, self.n_steps, x)
        lstm_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    def pred(self, seq_data):

        tf.reset_default_graph()   # 重置流图
        xtr = tf.placeholder("float", [None, self.n_steps, self.n_input])
        # ytr = tf.placeholder("float", [None, n_classes])
        weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden,
                                                self.n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
        # 获取预测值
        pred = self.RNN(xtr, weights, biases)

        saver1 = tf.train.Saver()
        saver2 = tf.train.Saver()
        saver3 = tf.train.Saver()
        saver4 = tf.train.Saver()
        saver5 = tf.train.Saver()
        # 创建会话，加载模型
        with tf.Session() as sess_1:
            saver1.restore(sess_1, 'predict_1min.ckpt')
        # 将当前数据代入模型，获得1-5分钟的分类类别
        # 7：  4：小涨   3：平稳   2：小跌   1：大跌
            pred_1 = sess_1.run(pred, feed_dict={xtr: seq_data})

        with tf.Session() as sess_2:
            saver2.restore(sess_2, "predict_2min.ckpt")
            pred_2 = sess_2.run(pred, feed_dict={xtr: seq_data})

        with tf.Session() as sess_3:
            saver3.restore(sess_3, "predict_3min.ckpt")
            pred_3 = sess_3.run(pred, feed_dict={xtr: seq_data})

        with tf.Session() as sess_4:
            saver4.restore(sess_4, "predict_4min.ckpt")
            pred_4 = sess_4.run(pred, feed_dict={xtr: seq_data})

        with tf.Session() as sess_5:
            saver5.restore(sess_5, "predict_5min.ckpt")
            pred_5 = sess_5.run(pred, feed_dict={xtr: seq_data})

        predict_1 = 7 - pred_1.argmax()
        predict_2 = 7 - pred_2.argmax()
        predict_3 = 7 - pred_3.argmax()
        predict_4 = 7 - pred_4.argmax()
        predict_5 = 7 - pred_5.argmax()

        return predict_1, predict_2, predict_3, predict_4, predict_5

    def data_use(self, datafr):  # 根据分钟数据进行矩阵计算

        test_data = DataFrame(datafr,
                              columns=['close', 'max', 'min', 'pos',
                                       'vol', 'open', 'askpr1', 'askvo1',
                                       'bidpr1', 'bidvo1'])
        # 取当前时刻往前n个单位长度的数据
        # test_data = test_data.fillna(0)
        test_new_data = test_data[-15:]
        # test_new_data = test_new_data.fillna(0)
        # where_are_nan = np.isnan(test_new_data)
        # where_are_inf = np.isinf(test_new_data)
        # test_new_data[where_are_nan] = 0
        # test_new_data[where_are_inf] = 0
        # 将数据变为数组形式并标准化
        data_new_array = np.array(test_new_data)
        min_max_scaler = preprocessing.MinMaxScaler()
        data_new_array = min_max_scaler.fit_transform(data_new_array)
        # test_new_data['close'] = (test_new_data['close'] - 2755.8) / (3640.4 - 2755.8)
        # test_new_data['max'] = (test_new_data['max'] - 2762.2) / (3646.0 - 2762.2)
        # test_new_data['min'] = (test_new_data['min'] - 2732.4) / (3640.2 - 2732.4)
        # test_new_data['pos'] = (test_new_data['pos'] - (-256.0)) / (211.0 - (-256.0))
        # test_new_data['vol'] = (test_new_data['vol'] - 1.0) / (1514.0 - 1.0)
        # test_new_data['open'] = (test_new_data['open'] - 2755.6) / (3644.8 - 2755.6)
        # test_new_data['askpr1'] = (test_new_data['askpr1'] - 2755.4) / (3640.2 - 2755.4)
        # test_new_data['askvo1'] = (test_new_data['askvo1'] - 1.0) / (48.0 - 1.0)
        # test_new_data['bidpr1'] = (test_new_data['bidpr1'] - 2759.0) / (3641.4 - 2759.0)
        # test_new_data['bidvo1'] = (test_new_data['bidvo1'] - 1.0) / (46.0 - 1.0)

        # data_new_array = np.array(test_new_data)
        # 生成n分钟序列
        seq_data = [data_new_array]
        seq_data = np.array(seq_data)

        return seq_data

    def pos_rec_concert(self):  # pos与pos_rec不一致时进行调整。
        if self.pos != self.pos_rec:
            self.writeCtaLog(u'调整pos_rec，由' +
                             str(self.pos_rec) + u'变为' + str(self.pos))
            self.pos_rec = self.pos

        if self.lastOrder is not None and self.lastOrder.status == u'未成交':
            self.cancelOrder(self.lastOrder.vtOrderID)
            self.lastOrder = None
            self.writeCtaLog(u'撤销上一单')

    def long_pos_sell(self, long_pos):  # 多仓的平仓考虑。
        if self.raise_count <= 1 and self.down_count >= 2:
            self.sell(self.bidprice1, 1)
            self.pos_rec -= 1
            self.writeCtaLog(u'sell_Price' + str(long_pos.bidPrice1))
            # self.zhisun_label = True
            # self.zhisun_bar = 0
            # self.trade_records.append([long_pos.datetime, self.price, u'sell'])

        # if long_pos.bidPrice1 < (1 - self.zhisun)*self.rec_price:#止损策略
        #     self.sell(self.bidprice1, 1)
        #     self.pos_rec -= 1
        #     self.writeCtaLog(u'多仓'+u'止损价'+str((1 - self.zhisun) * self.rec_price)+u'当前价'+str(long_pos.bidPrice1))
        #     self.zhisun_label = True
        #     self.zhisun_bar = 0
        #     self.trade_records.append([long_pos.datetime,self.price,u'sell'])

        # if long_pos.bidPrice1 > self.rec_price*(1+self.zhiying):##止盈策略
        #     self.sell(self.bidprice1, 1)
        #     self.pos_rec -= 1

        #     self.writeCtaLog(u'多仓'+u'止盈价'+str(self.rec_price*(1+self.zhiying))+u'当前价'+str(long_pos.bidPrice1))
        #     self.trade_records.append([long_pos.datetime,self.price,u'sell'])

    def short_pos_cover(self, short_pos):  # 持有空仓时的平仓考虑。
        if self.raise_count >= 2 and self.down_count <= 1:
            self.cover(self.askprice1, 1)
            self.pos_rec += 1
            self.writeCtaLog(u'cover_Price' + str(short_pos.askPrice1))

            # self.trade_records.append([short_pos.datetime, self.price, u'cover'])
        # if short_pos.askPrice1 > (1+self.zhisun)*self.rec_price: ##做空的时候，实时价格高于止损线
        #     self.cover(self.askprice1, 1)
        #     self.pos_rec += 1
        #     self.writeCtaLog(u'空仓'+u'止损价'+str((1 + self.zhisun) * self.rec_price)+u'当前价'+str(short_pos.askPrice1))
                    
        #     self.zhisun_label = True
        #     self.zhisun_bar = 0
        #     self.trade_records.append([short_pos.datetime,self.price,u'cover'])

        # if short_pos.askPrice1 < self.rec_price*(1-self.zhiying):#做空的时候，实时价格已经低于止盈线了
        #     self.cover(self.askprice1, 1)
        #     self.pos_rec += 1
        #     self.writeCtaLog(u'空仓'+u'止盈价'+str(self.rec_price * (1 - self.zhiying))+u'当前价'+str(short_pos.askPrice1))

        #     self.trade_records.append([short_pos.datetime,self.price,u'cover'])

    # def zhisun_ying_rate_update(self, own_pos):
    #     self.rec_price = float(self.rec_price + own_pos.askPrice1)/2

    #     if self.zhisun > 0:
    #         self.zhisun = self.zhisun + 0.0000008
    #     if self.zhiying > 0:
    #         self.zhiying = self.zhiying + 0.0000008#tick中和bar中更新止盈止损一致

# 每次止损平仓后，考虑停止5分钟进行操作。
# 因此设置zhisun_label作为是否能重新建仓的标志，以zhisun_bar作为停止的分钟时长的计数。

    def zhisun_set(self):
        if self.zhisun_label == True:
            self.zhisun_bar += 1
        if self.zhisun_bar >= 5:
            self.zhisun_bar = 0
            self.zhisun_label = False

    def vi_mem_record(self):
        '''
        用来对内存占用情况进行记录，输出
        '''
        mem_info = psutil.virtual_memory()
        return (psutil.Process(os.getpid()).memory_info().vms, psutil.Process(os.getpid()).memory_info().rss, mem_info.total, mem_info.percent) #vms虚拟内存
    # ----------------------------------------------------------------------

    def onOrder(self, order):
        """收到委托变化推送（必须由用户继承实现）"""
        # 对于无需做细粒度委托控制的策略，可以忽略onOrder
        pass

    # ----------------------------------------------------------------------
    def onTrade(self, trade):
        """收到成交推送（必须由用户继承实现）"""
        # 对于无需做细粒度委托控制的策略，可以忽略onOrder
        pass


##########################################################################

class OrderManagementDemo(CtaTemplate):
    """基于tick级别细粒度撤单追单测试demo"""

    className = 'OrderManagementDemo'
    author = u'用Python的交易员'

    # 策略参数
    initDays = 10   # 初始化数据所用的天数

    # 策略变量
    bar = None
    barMinute = EMPTY_STRING

    # 参数列表，保存了参数的名称
    paramList = ['name',
                 'className',
                 'author',
                 'vtSymbol']

    # 变量列表，保存了变量的名称
    varList = ['inited',
               'trading',
               'pos']

    # ----------------------------------------------------------------------
    def __init__(self, ctaEngine, setting):
        """Constructor"""
        super(OrderManagementDemo, self).__init__(ctaEngine, setting)

        self.lastOrder = None
        self.orderType = ''

    # ----------------------------------------------------------------------
    def onInit(self):
        """初始化策略（必须由用户继承实现）"""
        self.writeCtaLog(u'双EMA演示策略初始化')

        initData = self.loadBar(self.initDays)
        for bar in initData:
            self.onBar(bar)

        self.putEvent()

    # ----------------------------------------------------------------------
    def onStart(self):
        """启动策略（必须由用户继承实现）"""
        self.writeCtaLog(u'双EMA演示策略启动')
        self.putEvent()

    # ----------------------------------------------------------------------
    def onStop(self):
        """停止策略（必须由用户继承实现）"""
        self.writeCtaLog(u'双EMA演示策略停止')
        self.putEvent()

    # ----------------------------------------------------------------------
    def onTick(self, tick):
        """收到行情TICK推送（必须由用户继承实现）"""

        # 建立不成交买单测试单
        if self.lastOrder == None:
            self.buy(tick.lastprice - 10.0, 1)

        # CTA委托类型映射
        if self.lastOrder != None and self.lastOrder.direction == u'多' and self.lastOrder.offset == u'开仓':
            self.orderType = u'买开'

        elif self.lastOrder != None and self.lastOrder.direction == u'多' and self.lastOrder.offset == u'平仓':
            self.orderType = u'买平'

        elif self.lastOrder != None and self.lastOrder.direction == u'空' and self.lastOrder.offset == u'开仓':
            self.orderType = u'卖开'

        elif self.lastOrder != None and self.lastOrder.direction == u'空' and self.lastOrder.offset == u'平仓':
            self.orderType = u'卖平'

        # 不成交，即撤单，并追单
        if self.lastOrder != None and self.lastOrder.status == u'未成交':

            self.cancelOrder(self.lastOrder.vtOrderID)
            self.lastOrder = None
        elif self.lastOrder != None and self.lastOrder.status == u'已撤销':
            # 追单并设置为不能成交

            self.sendOrder(self.orderType, self.tick.lastprice - 10, 1)
            self.lastOrder = None

    # ----------------------------------------------------------------------
    def onBar(self, bar):
        """收到Bar推送（必须由用户继承实现）"""
        pass

    # ----------------------------------------------------------------------
    def onOrder(self, order):
        """收到委托变化推送（必须由用户继承实现）"""
        # 对于无需做细粒度委托控制的策略，可以忽略onOrder
        self.lastOrder = order

    # ----------------------------------------------------------------------
    def onTrade(self, trade):
        """收到成交推送（必须由用户继承实现）"""
        # 对于无需做细粒度委托控制的策略，可以忽略onOrder
        pass

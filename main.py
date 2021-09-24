# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:21:36 2020

@author: 10025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.reset_default_graph()

# ——————————————————导入数据——————————————————————
# 读入数据
data = pd.read_excel(r'C:\Programs\PyPrograms\LSTM_OG_09241511\aggregated.xls')
data = data.values
# 定义常量
rnn_unit = 10  # hidden layer units
input_size = 4  # 数据输入维度
output_size = 1  # 数据输出维度
lr = 0.0006  # 学习率
train_num=2960
total_num=3500


# 获取训练集
def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=train_num):  # 用前5800个数据作为训练样本
    print('get train data %d %d %d %d'%(batch_size, time_step, train_begin, train_end))
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
    print('std')
    print(normalized_train_data)
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :input_size]
        y = normalized_train_data[i:i + time_step, input_size, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(time_step=20, test_begin=train_num):  # 用5800之后数据作为测试样本
    print('get test data')
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :input_size]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, input_size]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :input_size]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, input_size]).tolist())
    return mean, std, test_x, test_y


# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, output_size]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[output_size, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(X):
    print('lstm')
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm(batch_size=80, time_step=15, train_begin=0, train_end=train_num):
    print('train lstm')
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    # 训练样本中，每次取15个
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    print(np.array(train_x).shape)
    print(batch_index)
    # 相当于每3个特征（embadding）,对于这些样本每次训练80
    pred, _ = lstm(X)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练200次
        for i in range(200):
            # 每次进行训练的时候，每个batch训练batch_size个样本
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})
            print(i, len(batch_index)-1, loss_)
            if i == 199:
                saver.save(sess, "model/stock2.ckpt")
                print("保存模型：%d" % i)
                for var in tf.trainable_variables():
                    print("Listing trainable variables ... ")
                    print(var)
                reader = tf.train.NewCheckpointReader("model/stock2.ckpt")
                global_variables = reader.get_variable_to_shape_map()
                for key in global_variables:
                    print("tensor_name: ", key)
                    print(reader.get_tensor(key))


with tf.variable_scope('train'):
    train_lstm()


# ————————————————预测模型————————————————————
def prediction(time_step=20):
    print('prediction')
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[input_size] + mean[input_size] #获得测试结果的均值
        test_predict = np.array(test_predict) * std[input_size] + mean[input_size]
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()


with tf.variable_scope('train', reuse=True):
    prediction()

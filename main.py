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
data = pd.read_excel('aggregated.xls',header = None)
data = data.values
batch_data=pd.read_excel('batch_train_60.xls',header = None)
batch_data=batch_data.values  # batch_data[i]表示训练集第i个batch的长度
batch_data_test=pd.read_excel('batch_test_60.xls',header = None)
batch_data_test=batch_data_test.values  # batch_data_test[i]表示测试集中第i个batch的长度
# 定义常量
rnn_unit = 10  # hidden layer units
input_size = 4  # 数据输入维度
output_size = 1  # 数据输出维度
lr = 0.0008  # 学习率
train_num=3000
total_num=3600
batch_size=60  # batch个数
step_max=60  # 一个batch的最大步数
train_times=100  # 训练轮次

# 由每个batch的大小bd，计算每个batch开始位置
def get_batch_index(bd):
    print('get batch index')
    ans=[]
    sum=0
    ans.append(sum)
    print(bd)
    for i in range(len(bd[0])):
        sum+=bd[0][i]
        ans.append(sum)
    return ans  # ans[i]是第i个batch开始位置（从0开始计数）

# with tf.variable_scope('train'):
#     batch_index = get_batch_index(batch_data)
#     batch_index_arr=np.array(batch_index)
#     i=0
#     a = batch_index_arr[i]
#     b = batch_index_arr[i + 1]
#     data_train = data[0:3000]
#     normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
#     x = normalized_train_data[0:2, :input_size]
#     print(x)


# 获取训练集
def get_train_data(train_begin=0, train_end=train_num):
    print('get train data %d %d' % (train_begin, train_end))
    batch_index = get_batch_index(batch_data)
    batch_index_arr=np.array(batch_index)
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
    print('std')
    print(normalized_train_data)
    train_x, train_y = [], []  # 训练集
    for i in range(len(batch_index)-1):
        x = normalized_train_data[batch_index[i]:batch_index[i+1], :input_size]
        y = normalized_train_data[batch_index[i]:batch_index[i+1], input_size, np.newaxis]
        # 用0补齐至step_max
        fill_x = []
        fill_y = []
        for t in range(input_size):
            fill_x.append(0)
        for t in range(output_size):
            fill_y.append(0)
        while len(x) < step_max:
            x.append(fill_x)
        while len(y) < step_max:
            y.append(fill_y)
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    print('size of batch_index, train_x, train_y')
    print(len(batch_index), len(train_x), len(train_y))
    print(len(train_x[1]))
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(test_begin=train_num):
    print('get test data')
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    batch_index_test=get_batch_index(batch_data_test)
    test_x, test_y = [], []
    for i in range(len(batch_index_test) - 1):
        x = normalized_test_data[batch_index_test[i]:batch_index_test[i+1], :input_size]
        y = normalized_test_data[batch_index_test[i]:batch_index_test[i+1], input_size]
        test_x.append(x.tolist())
        test_y.extend(y)
        # 对x和y补0
        fill_x = []
        fill_y = []
        for t in range(input_size):
            fill_x.append(0)
        for t in range(output_size):
            fill_y.append(0)
        while len(test_x[i]) < step_max:
            test_x[i].append(fill_x)
        t=batch_data_test[0][i]
        print(t)
        while t < step_max:
            test_y.extend(fill_y)
            t+=1
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
def lstm(X, isTrain = 1):
    print('lstm')
    batch_s = tf.shape(X)[0]  # batch个数
    step = tf.shape(X)[1]  # 一个batch内的最大行数
    w_in = weights['in']
    b_in = biases['in']
    input_lstm = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input_lstm, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_s, dtype=tf.float32)
    # print(batch_data[0])
    seq_len = batch_data_test[0]
    if isTrain:
        seq_len = batch_data[0]
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, sequence_length=seq_len, initial_state=init_state,
                                                 dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm(train_begin=0, train_end=train_num):
    print('train lstm')
    X = tf.placeholder(tf.float32, shape=[50, step_max, input_size])
    Y = tf.placeholder(tf.float32, shape=[50, step_max, output_size])
    # 获取训练样本
    batch_index, train_x, train_y = get_train_data(train_begin, train_end)
    print(np.array(train_x).shape)
    print(np.array(train_y).shape)
    print(batch_index)
    pred, _ = lstm(X,1)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练
        for i in range(train_times):
            # 每个batch训练
            for step in range(len(batch_index) - 1):
                # print(i,step)
                tx = train_x[step:step+1]
                ty = train_y[step:step+1]
                # print(np.array(tx).shape)
                # print(np.array(ty).shape)
                # _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                #                                                  Y: train_y[batch_index[step]:batch_index[step + 1]]})
                # _, loss_ = sess.run([train_op, loss], feed_dict={X: tx, Y: ty})
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})
            print(i, loss_)
            if i == train_times-1:
                saver.save(sess, "model/stock2.ckpt")
                print("保存模型：%d" % i)
                # for var in tf.trainable_variables():
                #     print("Listing trainable variables ... ")
                #     print(var)
                reader = tf.train.NewCheckpointReader("model/stock2.ckpt")
                # global_variables = reader.get_variable_to_shape_map()
                # for key in global_variables:
                #     print("tensor_name: ", key)
                #     print(reader.get_tensor(key))


with tf.variable_scope('train'):
    train_lstm()


# ————————————————预测模型————————————————————
def prediction():
    print('prediction')
    X = tf.placeholder(tf.float32, shape=[10, step_max, input_size])
    mean, std, test_x, test_y = get_test_data()
    print(test_x)
    print(np.array(test_y).shape)
    pred, _ = lstm(X,0)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: test_x})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[input_size] + mean[input_size]  # 获得测试结果的均值
        test_predict = np.array(test_predict) * std[input_size] + mean[input_size]
        test_predict=test_predict[:total_num-train_num]
        print(test_predict)
        # acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()


with tf.variable_scope('train', reuse=True):
    prediction()

# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from mnist import input_data
import numpy as np

# 常用变量以及前向传播函数
from mnist.mnist_leNet5_optimize import mnist_inference

"""
一个训练batch中的训练数据的个数, 数字越小越接近随机梯度下降, 反之越接近梯度下降
随机梯度下降 速度快,不一定能得到局部最优解
梯度下降速度慢, 至少能得到局部最优解
BATCH是折中做法
"""
BATCH_SIZE = 100

"""
学习率的设置, 指数衰减.
让模型在前期迅速接近最优解的参数
"""
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率

"""
解决过拟合问题, 限制权重大小, 使模型不能任意你和训练数据中的随机噪音
"""
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000  # 训练轮数

"""
提高模型的健壮性, 控制模型的更新速度.
越趋近于1 越稳定.
"""
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

# 模型保存的相关信息
MODEL_SAVE_PATH = "archive/"
MODEL_NAME = "mnist.ckpt"


def train(mnist):

    # 定义了从MNIST文件读取的测试图片和正确label的输入占位符
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,  # 样例的个数
                        mnist_inference.IMAGE_SIZE,  # 图片的尺寸
                        mnist_inference.IMAGE_SIZE,
                        mnist_inference.NUM_CHANNELS],  # 图片的深度, 对于RGB来说, 深度为3
                       name="x-input")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 调用前向传播过程
    y = mnist_inference.inference(x, True, regularizer)

    global_step = tf.Variable(trainable=False, initial_value=0)

    # 滑动平均模型的定义, 以及使用在所有可训练变量
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

    # 交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 总损失 = 交叉熵损失 + 正则化的权重
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    # 设置指数衰减学习率
    # LEARNING_RATE_BASE 基础学习率, 随着迭代进行, 更新变量时使用学习率在这个基础上递减.
    # mnist.train.num_examples / BATCH_SIZE 完成所有训练需要迭代的次数
    # LEARNING_RATE_DECAY 学习率衰减的速度
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    # 使用GradientDescentOptimizer优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss, global_step=global_step)

    # 在训练神经网络模型时, 每过一遍数据既要通过反向传播来更新神经网络模型的参数, 同时需要更新每一个参数的滑动平均值
    # 为了一次完成多个操作, TensorFlow提供了tf.control_dependencies 和 tf.group 两种机制, 下面两行程序和
    # train_op = tf.group(train_step, variables_averages_op) 是等价的.
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name="train")

    # 初始化持久化类
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        # 在训练的过程中不再测试模型在验证数据上面的表现, 验证和测试的过程将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            # 适应卷及神经网络
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNELS))

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出当前的训练情况, 这里只输出了模型在当前的训练batch上的损失函数大小
                # 通过损失函数的大小可以大概的了解训练的情况
                # 在验证数据集上的正确率信息会有一个单独的程序来完成
                print("After %d training steps, loss on training batch is %g" %(step, loss_value))
                # 保存当前的模型, 注意这里给出了global_step参数, 这样可以让每个被保存的模型的文件名末尾加上训练的轮数
                # 比如"mnist.ckpt-1000"表示训练1000轮以后的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_Data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
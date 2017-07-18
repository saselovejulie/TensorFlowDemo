# -*- coding: utf-8 -*-
import tensorflow as tf

# MNIST数据集相关配置
INPUT_NODE = 784  # 代表一张图片的784个像素点作为输入节点
IMAGE_SIZE = 28  # 图片的长度像素个数
OUTPUT_NODE = 10  # 0-9个数字的10分类问题, 所以输出是一个10节点的矩阵
NUM_CHANNELS = 1  # 图像的深度, 一般指颜色的组成. 黑白深度为1, 一般彩色深度为3(RGB)
NUM_LABELS = 10

# 第一层卷积层的配置
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第一层卷积层的配置
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接的节点个数
FC_SIZE = 512


def inference(input_tensor, train, regularizer):
    """
    定义神经网络的前向传播过程, 这里添加了一个新的参数train用于区别训练过程和测试过程.
    在这个程序中将用到dropout方法, dropout可以进一步提升模型的可靠性并防止过拟合,
    dropout只在测试过程中使用
    :param input_tensor:
    :param train: 训练中加入 dropout方法解决过拟合
    :param regularizer: 正则化函数, tf的L1或L2
    :return:
    """
    with tf.variable_scope("layer1-conv1"):
        """
        声明卷积层的第一层的变量并实现前向传播过程.这个过程和6.3.1小节中的介绍一致.
        通过使用不同的命名空间来隔离不同层的变量, 这可以让每一层的变量命名只需要考虑当前层的作用, 不用担心重命名问题
        定义的卷积层输入为28*28*1,因为卷积层使用了全0填充, 所以输出为28*28*32.
        """
        conv1_weights = tf.get_variable("weights",
                                        [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))

        conv1_biases = tf.get_variable("biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.1))
        # 使用边长为5, 深度为32的过滤器. 过滤器的移动步长为1. 且用全0补充.
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层前向传播过程, 这里用到最大池化层, 池化曾过滤器的边长为2.
    # 使用全0填充切步长为2, 这一层数输入是上一层输出, 也就是28*28*32的矩阵, 输出位14*14*32
    with tf.variable_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 第三层卷积层的变量
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weights",
                                        [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases", [CONV2_DEEP], initializer=tf.constant_initializer(0.1))
        # 边长5, 深度64过滤器, 步率1
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层池化传播, 输入14*14*64, 输出7*7*64
    with tf.variable_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 将第四层池化层的输出转化为第五层全连接层的输入格式, 第四层的输出位7*7*64的矩阵,
    # 然而第五层全连接层需要的输入格式为向量, 所以在这里需要将这个7*7*64的矩阵拉直为一个向量.
    # pool2.get_shape函数可以得到第四层的输出矩阵的维度而不需要手工计算
    # 注意, 因为每一层的神经网络的输入输出都为一个batch矩阵, 所以这里得到的维度也包含了一个batch的数据的个数
    pool_shape = pool2.get_shape().as_list()

    # 计算矩阵拉直成向量之后的长度
    # pool_shape[0]为一个batch中数据的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    # 通过tf.reshape函数将第四层的输出变成一个batch向量
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 第五层全连接层的变量声明并实现前向传播过程, 这一层的输入是拉直以后的一组向量.
    # 向量的长高度为3136, 输出的是一组长度为512的向量. 和之前介绍的全连接层一致, 但是引入了dropout.
    # dropout在训练时会随机将部分节点的输出为0, dropout可以避免过分拟合问题, 从而使得模型在测试数据上的效果更好.
    # dropout一般只在全连接层使用, 卷积和池化不用.
    with tf.variable_scope("layer5-fc1"):
        fc1_weights = tf.get_variable("weights", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要增加正则化
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc1_weights))

        fc1_biases = tf.get_variable("biases", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 声明
    with tf.variable_scope("layer6-fc2"):
        fc2_weights = tf.get_variable("weights", [FC_SIZE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc2_weights))

        fc2_biases = tf.get_variable("biases", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit


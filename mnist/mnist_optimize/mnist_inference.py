# -*- coding: utf-8 -*-
import tensorflow as tf

# MNIST数据集相关配置
INPUT_NODE = 784  # 代表一张图片的784个像素点作为输入节点
OUTPUT_NODE = 10  # 0-9个数字的10分类问题, 所以输出是一个10节点的矩阵

# 配置神经网络参数
LAYER1_NODE = 500  # 隐藏层节点数, 这里使用只有一个隐藏层的网络结构作为样例, 这个隐藏层有500个节点


def get_weight_variable(shape, regularizer):
    """
    通过tf.get_variable获取变量, 在训练神经网络的时候会创建这些变量. 在测试时会通过保存的模型加载这些变量的取值
    而且更加方便的是, 因为可以在变量加载的时候将滑动平均变量重命名, 所以可以直接通过同样的名字在训练时使用变量自身.
    而在测试时使用变量的滑动平均值. 在这个函数中也会将变量的正则化损失加入损失集合
    :param shape: 权重的矩阵结构
    :param regularizer: 平均滑动模型(可选)
    :return: 权重
    """
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    """
    当给出了正则化生成函数时, 将当前的变量的正则化损失加入名字为losses的集合.
    在这里使用了add_to_collection函数将一个张量加入一个集合, 而这个集合的名称为losses.
    这是一个自定义集合, 不在TensorFlow的自动管理的列表之中
    """
    if regularizer is not None:
        tf.add_to_collection("losses", regularizer(weights))

    return weights


def inference(input_tensor, regularizer):
    """
    定义神经网络的前向传播过程, 返回双层神经网络模型的前向传播结果
    :param input_tensor:
    :param regularizer:
    :return:
    """
    # 声明第一层的神经网络的变量并完成前向传播过程
    with tf.variable_scope("layer1"):
        # 这里通过get_variable或是Variable没有本质区别, 因为在训练或是测试中
        # 没有在同一个训练中多次调用这个函数. 如果在同一个程序中多次调用, 在第一次调用之后需要将reuse参数设置为True
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        print(input_tensor)
        print(weights)
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 声明第二层神经网络变量并完成前向传播过程
    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2

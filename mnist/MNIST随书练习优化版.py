import tensorflow as tf
import input_data

"""
激活函数+隐藏层+交叉熵+softmax+正则化+滑动平均模型+指数化学习率
可优化提升的地方: 卷及神经网络 + 交叉验证法
"""

# MNIST数据集相关配置
INPUT_NODE = 784  # 代表一张图片的784个像素点作为输入节点
OUTPUT_NODE = 10  # 0-9个数字的10分类问题, 所以输出是一个10节点的矩阵

# 配置神经网络参数
LAYER1_NODE = 500  # 隐藏层节点数, 这里使用只有一个隐藏层的网络结构作为样例, 这个隐藏层有500个节点

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


def inference(input_tonser, avg_class, weights1, biases1, weights2, biases2):
    """
    一个辅助函数,给定神经网络的输入和所有参数, 计算神经网络的前向传播结果.
    定义了一个使用ReLU激活函数的三层全连接的神经网络. 通过加入隐藏层实现了多层网络结构.
    通过ReLU实现了去线性化, 在这个函数中也支持传入用于计算参数平均值的类, 方便使用滑动平均模型.
    :param input_tonser: 输入张量
    :param avg_class: 滑动平均值计算器
    :param weights1: 权重1
    :param biases1: 偏移量1
    :param weights2: 权重2
    :param biases2: 偏移量2
    :return: tf前向计算传播模型
    """
    # 如果没有传入滑动模型, 采用当前权重和偏移量进行计算
    if avg_class is None:
        # 计算隐藏层的前向传播结果, 使用了ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tonser, weights1) + biases1)
        """
        输出前向传播结果, 因为在计算损失函数时会一并计算softmax函数, 所以这里不需要加入激活函数. 而且不加入激活函数不会影响softmax的预测结果.
        因为预测是使用的是不同类别对应的节点输出值的相对大小, 有没有softmax层对最后的分类结果的计算没有影响.
        于是在计算整个神经网络前向传播时可以不加入最后的softmax层
        """
        return tf.matmul(layer1, weights2) + biases2
    else:
        # avg_class处理权重和偏移量的滑动平均值
        layer1 = tf.nn.relu(
            tf.matmul(input_tonser, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    """
    训练模型的过程
    :param mnist: mnist函数
    :return:
    """
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数, 784个像素分配给500个神经网络节点进行处理
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    # 偏移量固定0.1
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数, 500个神经网络节点的输出值处理为0-9个数字, 10分类问题
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算当前参数下的前向传播结果, 暂不使用滑动平均值增加健壮性
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量, 这个变量不需要计算滑动平均值, 所以指定这个变量为不可训练变量. 训练神经网络中一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数变量, 初始化滑动平均类. 加快早期的变量更新速度
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均, 辅助变量如global_step不需要.
    # tf.trainable_variables返回的就是图上集合GraphKeys.TRAINABLE_VARIABLES中的元素, 这个集合参数就是所有没有指定trainable=False的变量
    variables_average_op = variable_average.apply(tf.trainable_variables())

    # 计算使用了滑动平均值后的前向传播结果, 滑动平均不会改变变量本身的取值, 而是会维护一个影子变量来记录其滑动平均值.
    # 所以当需要使用这个滑动平均值的时候需要明确调用average函数
    average_y = inference(x, variable_average, weights1, biases1, weights2, biases2)

    # 计算交叉熵作为刻画预测值和真实值之间的差距的损失函数, 这里使用了TensorFlow的sparse_softmax_cross_entropy_with_logits函数来计算交叉熵
    # 当分类问题只有一个正确答案时可以使用此函数来加速交叉熵的计算.
    # 第一参数是神经网络不包括softmax的前向传播结果. 第二个是训练数据的正确答案.
    # 因为正确答案时一个[10,1]的矩阵, 需要用argmax函数筛选出最大值作为正确答案对应的类别编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算模型的正则化损失, 一般只计算神经网络边上的权重的正则化损失, 而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失 = 交叉熵损失 + 正则化损失的和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减学习率
    # LEARNING_RATE_BASE 基础学习率, 随着迭代进行, 更新变量时使用学习率在这个基础上递减.
    # mnist.train.num_examples / BATCH_SIZE 完成所有训练需要迭代的次数
    # LEARNING_RATE_DECAY 学习率衰减的速度
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    # 使用GradientDescentOptimizer优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时, 每过一遍数据既要通过反向传播来更新神经网络模型的参数, 同时需要更新每一个参数的滑动平均值
    # 为了一次完成多个操作, TensorFlow提供了tf.control_dependencies 和 tf.group 两种机制, 下面两行程序和
    # train_op = tf.group(train_step, variables_averages_op) 是等价的.
    with tf.control_dependencies([train_step, variables_average_op]):
        train_op = tf.no_op(name='train')

    # 检验使用了滑动平均模型的前向传播结果是否正确. tf.argmax(average_y, 1) 计算每一个样例的预测答案, 其中average_y是一个batch_size*10的二维数组.
    # 每一行表示一个样例的前向传播结果, tf.argmax的第二个参数1表示选取最大值的操作尽在第一维度进行, 也就是说只在每一行选取最大值对应的下标. 于是得到的结果是
    # 一个长度为batch的一维数组, 这个一维数组的值就表示了每个样例对应数字的识别结果.tf.equal判断两个张量的每一维是否相等, 如果想等返回True
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # 这个运算首先将一个布尔型的值转化为一个实数型, 然后计算平均值. 这个平均值就是模型在这一组的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        # 准备验证数据, 一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        # 准备测试数据, 在真实应用中, 这部分数据在训练时是不可见的, 这个数据只是作为模型优劣的最后评判标准
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        # 迭代进行训练神经网络模型
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果.
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据集上的效果, 因为MNIST的数据集比较小, 所以一次可处理所有的验证数据.
                # 为了方便计算,本样例程序没有将验证数据划分为更小的batch,
                # 当神经网络的模型比较复杂活者验证数据比较大时, 太大的batch会导致计算时间过长甚至发生内存溢出
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy using average model is %g" % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps, test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets(train_dir="MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

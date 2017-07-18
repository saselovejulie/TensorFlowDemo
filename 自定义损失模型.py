import tensorflow as tf
from numpy.random import RandomState

"""

"""

# 每一次分析8组数据(步长8)
batch_size = 8

# 2个输入节点.
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# 1个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
# 权重. 接受2个输入值, 输出一个数据值. 前向传播过程, 加权和.
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
# 预测值为x和w1矩阵相乘
y = tf.matmul(x, w1)

# 定义损失函数使得预测少了的损失大，于是模型应该偏向多的方向预测。
loss_less = 10
loss_more = 1
# tf.greater 对比2个矩阵相同位置的数据, 第一个更大则为True. 返回一个矩阵
# tf.where 第一个表达式为True则返回第二个参数, 否则第三个
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
# 0.001梯度 最小化损失
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
# 128行 2列随机数. [0-1) 之间
X = rdm.rand(128, 2)
# 设置正确的回归值Y为2个输入+噪音. 噪音一般为均值为0的小量
# for (x1, x2) in X:
#     print("x1 is : %f, x2 is : %f " % (x1, x2))
Y = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]

print(Y)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % 128
        end = (i*batch_size) % 128 + batch_size
        # X为输入变量,2个属性 Y为正确的回归值
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            print("After %d training step(s), w1 is: " % i)
            print(sess.run(w1), "\n")
    print("Final w1 is: \n", sess.run(w1))
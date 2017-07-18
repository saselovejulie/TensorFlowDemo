import input_data
import tensorflow as tf

"""
Mnist初级版本, 无隐藏层,
无激活函数, 无正则优化, 无指数衰减学习率, 无滑动平均模型
正确率 91~92%
"""

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 用于得到传递进来的真实的训练样本, 第一维度不限制数量代表不限制图片张数, 第二维度784个元素代表每个图片784个点. 二阶矩阵
x = tf.placeholder("float", [None, 784])
# 在此张量里的每一个元素，都表示某张图片里的某个像素的强度值，值介于0和1之间
# 二阶矩阵,784元素在一维, 10元素在二维.784代表一张图片784个点, 10代表每个点的颜色强度
W = tf.Variable(tf.zeros([784, 10]), name='weights')
# 偏置量（bias），因为输入往往会带有一些无关的干扰量。
# [0,0,0,0,0,0,0,0,0]
b = tf.Variable(tf.zeros([10]), name='biases')

# 将W乘x加b带入softMax.矩阵乘法
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 交叉熵计算,
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(4000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()

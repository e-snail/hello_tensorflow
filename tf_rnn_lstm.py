from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 训练参数
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# 网络参数
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# 初始权值
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# 核心模型
def RNN(x, weights, biases):
    # unstack: 将x 在列维度上(axis=1) 拆分分量, 列分量的有timesteps行, timesteps可以根据shape推算出来，不用指定
    print("shape(x) =", tf.shape(x))
    x = tf.unstack(x, num=timesteps, axis=1)
    print("shape(x) =", tf.shape(x))
    # print("shape(weights) =", tf.shape(weights))
    # print("shape(biases) =", tf.shape(biases))

    # rnn.BasicLSTMCell()的参数
    # num_hidden    在LSTM cell中unit 的数目
    # forget_bias   添加到遗忘门中的偏置
    # 参数3  输入到LSTM cell 中输入的维度。默认等于 num_units
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# 定义损失和优化函数
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for step in range(1, training_steps+1):
        # batch_x [128x784] --> [128, 28, 28]  batch_y [128x10] 
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))

        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

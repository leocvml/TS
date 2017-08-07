import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("/tmp/data/" , one_hot = True)

training_epochs = 50

learning_rate = 0.01
batch_size = 100
display_step = 1


x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
evidence = tf.matmul(x,W) + b

act = tf.nn.softmax(evidence)



cross_entropy = y*tf.log(act)
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy,reduction_indices = 1))
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

avg_set = []
epoch_set = []
init =tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i  in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(opt,feed_dict ={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost,feed_dict={x: batch_xs, y: batch_ys}) / total_batch

        if epoch % display_step == 0:
            print("Epoch:",'%04d' %(epoch+1),"cost = ","{:.9f}".format(avg_cost))
        avg_set.append(avg_cost)
        epoch_set.append(epoch+1)
    print("training finish")

    correct_prediction = tf.equal(tf.argmax(act,1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print("model acc",acc.eval({x: mnist.test.images, y: mnist.test.labels}))



    plt.plot(epoch_set, avg_set, 'o',label = 'Logist Regression')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

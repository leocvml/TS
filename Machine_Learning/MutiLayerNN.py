import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("/tmp/data/" , one_hot = True)

learining_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1

n_hidden_1 = 256
n_hidden_2 = 256

n_input =784
n_classes = 10

x = tf.placeholder("float",[None, n_input])
y = tf.placeholder("float",[None,n_classes])

h = tf.Variable(tf.random_normal([n_input,n_hidden_1]))
bias_1 = tf.Variable(tf.random_normal([n_hidden_1]))
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,h),bias_1))


w = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
bias_2 = tf.Variable(tf.random_normal([n_hidden_2]))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,w),bias_2))


output = tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
bias_output = tf.Variable(tf.random_normal([n_classes]))
output_layer = tf.matmul(layer_2,output) + bias_output

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_layer, labels = y))

optimizer = tf.train.AdamOptimizer(learining_rate).minimize(cost)
print("OK4")
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
            sess.run(optimizer,feed_dict ={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost,feed_dict={x: batch_xs, y: batch_ys}) / total_batch

        if epoch % display_step == 0:
            print("Epoch:",'%04d' %(epoch+1),"cost = ","{:.9f}".format(avg_cost))
        avg_set.append(avg_cost)
        epoch_set.append(epoch+1)
    print("training finish")

    correct_prediction = tf.equal(tf.argmax(output_layer,1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print("model acc",acc.eval({x: mnist.test.images, y: mnist.test.labels}))



    plt.plot(epoch_set, avg_set, 'o',label = 'Logist Regression')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

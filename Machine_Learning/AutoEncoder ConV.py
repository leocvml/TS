from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 2
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters

n_input = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features


def conv2d(img,w,b):
    return tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1,2,2,1], padding='SAME'),b))

def deconv2d(x,W,output_shape):
    return tf.nn.conv2d_transpose(x,W,output_shape, strides=[1,2,2,1], padding = 'SAME')

def max_pool(img,k):
    return tf.nn.max_pool(img, ksize=[1,k,k,1], strides=[1,k,k,1], padding='Same')

def encoder(x):
    #5x5conv 1input 32output
    wc1 = tf.Variable(tf.random_normal([5,5,1,32]))
    bc1 = tf.Variable(tf.random_normal([32]))

    #5x5conv 32input 64outputs
    wc2 = tf.Variable(tf.random_normal([5,5,32,64]))
    bc2 = tf.Variable(tf.random_normal([64]))

    '''
    #fully conncet 28*28*64input 1024output
    wd1 = tf.Variable(tf.random_normal([7*7*64, 1024]))
    bd1 = tf.Variable(tf.random_normal([1024]))

    #1024input 10output
    wout = tf.Variable(tf.random_normal([1024,128]))
    bout = tf.Variable(tf.random_normal([128]))
    '''

    #construct model
    _X = tf.reshape(x, shape=[-1,28,28,1])
    conv1 = conv2d(_X,wc1,bc1)
    conv2 = conv2d(conv1,wc2,bc2)
    '''
    #reshape conv2 output to fit dense layer input
    dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1,wd1),bd1))
    pred = tf.add(tf.matmul(dense1,wout),bout)
    '''
    print("code layer shape : %s" % conv2.get_shape())

    return conv2



def decoder(ori_img,code):
     
    w_dc1 = tf.Variable(tf.random_normal([5,5,32,64]))
    b_dc1 = tf.Variable(tf.random_normal([1]))
    output_shape_d_conv1 = tf.stack([tf.shape(ori_img)[0], 14, 14, 32])
    h_d_conv1 = tf.nn.sigmoid(deconv2d(code,w_dc1,output_shape_d_conv1))

    w_dc2 = tf.Variable(tf.random_normal([5,5,1,32]))
    b_dc2 = tf.Variable(tf.random_normal([32]))
    output_shape_d_conv2 = tf.stack([tf.shape(ori_img)[0], 28, 28, 1])
    h_d_conv2 = tf.nn.sigmoid(deconv2d(h_d_conv1,w_dc2,output_shape_d_conv2))

    print("reconstruct layer shape : %s" % h_d_conv2.get_shape())
    return h_d_conv2



# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
_X = tf.reshape(X, shape=[-1,28,28,1])
# Construct model
encoder_op = encoder(X)
decoder_op = decoder(X,encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = _X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_pred - y_true, 2))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)) ,cmap='gray')
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)),cmap = 'gray')
    f.show()
    '''
    plt.draw()
    plt.waitforbuttonpress()
    '''

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters

n_input = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding = 'SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, 784])
x_origin = tf.reshape(x, [-1, 28, 28, 1])

W_e_conv1 = tf.Variable(tf.random_normal([5,5,1,16]))
b_e_conv1 = tf.Variable(tf.random_normal([16]))
h_e_conv1 = tf.nn.relu(tf.add(conv2d(x_origin, W_e_conv1), b_e_conv1))

W_e_conv2 = tf.Variable(tf.random_normal([5,5,16,32]))
b_e_conv2 = tf.Variable(tf.random_normal([32]))
h_e_conv2 = tf.nn.relu(tf.add(conv2d(h_e_conv1, W_e_conv2), b_e_conv2))

code_layer = h_e_conv2
print("code layer shape : %s" % h_e_conv2.get_shape())

W_d_conv1 = tf.Variable(tf.random_normal([5,5,16,32]))
b_d_conv1 = tf.Variable(tf.random_normal([1]))
output_shape_d_conv1 = tf.stack([tf.shape(x)[0], 14, 14, 16])
print(x.get_shape()[0])
h_d_conv1 = tf.nn.relu(deconv2d(h_e_conv2, W_d_conv1, output_shape_d_conv1))

W_d_conv2 = tf.Variable(tf.random_normal([5,5,1,16]))
b_d_conv2 = tf.Variable(tf.random_normal([1]))
output_shape_d_conv2 = tf.stack([tf.shape(x)[0], 28, 28, 1])
print(x.get_shape()[0])
h_d_conv2 = tf.nn.relu(deconv2d(h_d_conv1, W_d_conv2, output_shape_d_conv2))

x_reconstruct = h_d_conv2
print("reconstruct layer shape : %s" % x_reconstruct.get_shape())

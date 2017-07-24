import tensorflow as tf
from matplotlib import pyplot as plt
from random import randint
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
'''
input_X= tf.placeholder(tf.float32, [None, 784])
Weight_1 = tf.Variable(tf.zeros([784, 10]))
bios_1 = tf.Variable(tf.zeros([10]))
out = tf.nn.softmax((tf.matmul(input_X,Weight_1)+bios_1))
out_label = tf.placeholder(tf.float32, [None, 10])
'''
input_X= tf.placeholder(tf.float32, [None, 784])
Weight_1 = tf.Variable(tf.zeros([784, 10]))
bios_1 = tf.Variable(tf.zeros([10]))
out = tf.nn.softmax((tf.matmul(input_X,Weight_1)+bios_1))
# Add ops to save and restore all the variables.
saver = tf.train.Saver()



# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/Users/leo/Desktop/tensorflow/modelSRstroe/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  num = randint(0, mnist.test.images.shape[0])
  img = mnist.test.images[num]
  classification = sess.run(tf.argmax(out, 1), feed_dict={input_X: [img]})
  plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
  plt.show()
  print ('NN predicted', classification[0])









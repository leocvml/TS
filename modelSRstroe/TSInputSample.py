from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
print("OK")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("OK")
input_X= tf.placeholder(tf.float32, [None, 784])
Weight_1 = tf.Variable(tf.zeros([784, 10]))
bios_1 = tf.Variable(tf.zeros([10]))
out = tf.nn.softmax((tf.matmul(input_X,Weight_1)+bios_1))


out_label = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(out_label * tf.log(out), reduction_indices=[1]))


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver = tf.train.Saver()


count = 0
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={input_X: batch_xs, out_label: batch_ys})
  #print("count:",count)
  count += 1
  



correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(out_label,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={input_X: mnist.test.images, out_label: mnist.test.labels}))
save_path = saver.save(sess, "/Users/leo/Desktop/tensorflow/MNIST_Weight/mnist")
print("Model saved in file: %s" % save_path)
#print(len(mnist.test.images[10]))



from matplotlib import pyplot as plt
from random import randint
num = randint(0, mnist.test.images.shape[0])
img = mnist.test.images[num]
classification = sess.run(tf.argmax(out, 1), feed_dict={input_X: [img]})
plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
plt.show()
print ('NN predicted', classification[0])

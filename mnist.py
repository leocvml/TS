from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
print("OK")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("OK")
input_X= tf.placeholder(tf.float32, [None, 784])
Weight_1 = tf.Variable(tf.zeros([784, 5]))
Weight_2 = tf.Variable(tf.zeros([5, 10]))
bios_1 = tf.Variable(tf.zeros([5]))
bios_2 = tf.Variable(tf.zeros([10]))

out_1 = tf.nn.softmax((tf.matmul(input_X,Weight_1)+bios_1))
out = tf.nn.softmax((tf.matmul(out_1,Weight_2)+bios_2))
#out = tf.nn.softmax((tf.matmul((tf.matmul(input_X, Weight_1) + bios_1),Weight_2)) + bios_2)


out_label = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(out_label * tf.log(out), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
print("OK")
count = 0
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={input_X: batch_xs, out_label: batch_ys})
  print("count:",count)
  count += 1
  



correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(out_label,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={input_X: mnist.test.images, out_label: mnist.test.labels}))

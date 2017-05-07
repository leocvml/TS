import tensorflow as tf

hello_constant = tf.constant('Hello World!')
sess = tf.Session()
print(sess.run(hello_constant))


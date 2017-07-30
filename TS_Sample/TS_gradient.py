import tensorflow as tf

x = tf.placeholder(tf.float32)

y = 2 * x * x + 5 * x

var_grad = tf.gradients(y,x)  #微分

with tf.Session() as session:
    var_grad_val = session.run(var_grad , feed_dict = { x:1 }) # x input 
    print(var_grad_val)

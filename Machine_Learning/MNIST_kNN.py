import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/" , one_hot = True)

train_pixels,train_list_values = mnist.test.next_batch(9900)
test_pixels ,test_list_values = mnist.test.next_batch(100)

train_pixel_tensor = tf.placeholder(tf.float32 , [None , 784])
test_pixel_tensor = tf.placeholder(tf.float32 , [784])


#  root( (train^2 - test^2) )
distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor,tf.negative(test_pixel_tensor))) , reduction_indices = 1)

pred = tf.arg_min(distance , 0)   # k = 1

accuracy = 0

init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    for i in range(len(test_list_values)):
        nn_index  = sess.run(pred,feed_dict={train_pixel_tensor:train_pixels,
                                            test_pixel_tensor:test_pixels[i,:]})   #find min distance index
        
        print("Test %d" , i , "Predict Class: " , np.argmax(train_list_values[nn_index]),
                  "True :", np.argmax(test_list_values[i]))
    
   
        if np.argmax(train_list_values[nn_index]) == np.argmax(test_list_values[i]):
            accuracy += 1./len(test_pixels)
    print("Result: " , accuracy)



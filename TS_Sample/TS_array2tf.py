import tensorflow as tf
import numpy as np

mat1 = np.array([(2,2,2),(2,2,2),(2,2,2)],dtype = 'int32')
mat2 = np.array([(1,1,1),(1,1,1),(1,1,1)],dtype = 'int32')

print("mat1" , mat1)
print("mat2" , mat2)

mat1 = tf.constant(mat1)
mat2 = tf.constant(mat2)

print("mat1" , mat1)
print("mat2" , mat2)

mat_pro = tf.matmul(mat1 , mat2)
mat_sum = tf.add(mat1 , mat2)

print("mat_pro" , mat_pro)
print("mat_sum" , mat_sum)

with tf.Session() as sess:
    result1 = sess.run(mat_pro)
    result2 = sess.run(mat_sum)

print(result1)
print(result2)

import tensorflow as tf
import numpy as np
from matplotlib import *
import matplotlib.image as mp_image
import matplotlib.pyplot as plt


filename = "ethz.jpg"
input_img = mp_image.imread(filename)
print("input dim={}".format(input_img.ndim))
print("input shape={}".format(input_img.shape))
plt.imshow(input_img)
  

    #slice img 2 tf
my_img = tf.placeholder("uint8",[None,None,3])
slice_img = tf.slice(my_img , [10,0,0],[311,-1,-1])
with tf.Session() as session:
    result = session.run(slice_img , feed_dict = {my_img : input_img})
    print(result.shape)
plt.imshow(result)
    

    #TRANSPOE
x = tf.Variable(input_img , name = 'x')
model = tf.global_variables_initializer()
with tf.Session() as session:
    x = tf.transpose(x , perm = [1,0,2])
    session.run(model)
    result = session.run(x)
plt.imshow(result)
plt.show()

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
mnist_images = input_data.read_data_sets("MNIST_data",one_hot = False)
pixels , real_values = mnist_images.train.next_batch(10)
print("list of values loaded " , real_values)
example_to_visualize = 5
print("element N" + str(example_to_visualize + 1) + "of the list poltted")
    
image = pixels[example_to_visualize , :]
image = np.reshape(image,[28,28])
plt.imshow(image, cmap='gray')
plt.show()

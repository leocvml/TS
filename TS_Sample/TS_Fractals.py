import tensorflow as tf
import numpy as np
import matplot.pyplot as plt

Y , X = np.mgrid[-1.3:1.3:0.005 , -2:1:0.005]
Z = X + 1j * Y
c = tf.constant (Z.astype(np.complex64))

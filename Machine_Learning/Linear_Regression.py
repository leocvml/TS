import numpy as np
import matplotlib.pyplot as plt
number_of_points = 500

x_point = []
y_point = []

a = 0.22
b = 0.78

for i in range(number_of_points):
    x = np.random.normal(0.0 , 0.5)
    y = a*x + b + np.random.normal(0.0 , 0.1)
    x_point.append(x)
    y_point.append(y)
'''
plt.plot(x_point , y_point , 'o' , label = 'Input Data')
plt.legend()
plt.show()
'''

import tensorflow as tf

A = tf.Variable(tf.random_uniform ([1] , -1.0 , 1.0))  #tf.random_uniform ([1] , -1.0 , 1.0)
b = tf.Variable(tf.zeros([1]))
y = A * x_point + b


cost_function = tf.reduce_mean(tf.square(y - y_point))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cost_function)

model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    for step in range(0,200):
        session.run(train)
        if(step % 5 == 0):
            plt.plot(x_point , y_point , 'o' , label='step = {}'.format(step))
            plt.plot(x_point,session.run(A)*x_point + session.run(b))
            plt.legend()
            plt.show()
            print(session.run(A),session.run(b))



import tensorflow as tf
import numpy as np
import math,random
import matplotlib.pyplot as plt

# data generation
NUM_point = 1000
np.random.seed(NUM_point)

function_2_learn = lambda x: np.cos(x) + 0.1*np.random.randn(*x.shape)

all_x = np.float32(np.random.uniform(-2*math.pi, 2*math.pi,(1, NUM_point))).T
np.random.shuffle(all_x)
train_size = int(900)

x_training = all_x[:train_size]
y_training = function_2_learn(x_training)

x_validation = all_x[train_size:]
y_validation = function_2_learn(x_validation)

plt.figure(1)
plt.scatter(x_training,y_training, c='blue', label='train')
plt.scatter(x_validation,y_validation,c='red', label='validation')
plt.show()



layer_1_neurons = 10
batch_size = 10
NUM_EPOCHS = 1500



X = tf.placeholder(tf.float32, [None, 1], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

w_h = tf.Variable(tf.random_uniform([1, layer_1_neurons],minval=-1,maxval=1,dtype=tf.float32))
b_h = tf.Variable(tf.zeros([1, layer_1_neurons],dtype=tf.float32))

h = tf.nn.sigmoid(tf.matmul(X,w_h) + b_h)

w_o = tf.Variable(tf.random_uniform([layer_1_neurons,1],minval=-1, maxval=1,dtype=tf.float32))
b_o = tf.Variable(tf.zeros([1,1],dtype = tf.float32))

model =  tf.matmul(h,w_o) + b_o

train_op = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(model - Y))  # l2_loss = ((model - Y)**2) / 2

sess = tf.Session()
sess.run(tf.global_variables_initializer())
error = []

for i in range(NUM_EPOCHS):
    for start,end in zip(range(0,len(x_training),batch_size),range(batch_size,len(x_training), batch_size)):
        sess.run(train_op,feed_dict={X: x_training[start:end], Y: y_training[start:end]})
    cost = sess.run(tf.nn.l2_loss(model - y_validation),feed_dict={X:x_validation})

    error.append(cost)
    if (i%100 == 0):
        print("epoch %d, cost = %g" %(i,cost))
plt.plot(error,label = "MLP , function approximation")
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend()
plt.show()
        
    














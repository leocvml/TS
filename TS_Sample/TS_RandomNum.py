import tensorflow as tf
import matplotlib.pyplot as plt

def uniform_dist():
    
# 均勻分布
    #random_uniform (shape , minval , maxval , dtype ,seed ,name)
    uniform = tf.random_uniform([100] , minval = 0 , maxval = 1 , dtype = tf.float32)
    sess = tf.Session()

    with tf.Session() as session:
        print(uniform.eval())
        plt.hist(uniform.eval(),normed = True)
        plt.show()


def normal_dist():
    norm = tf.random_normal([100] , mean = 0 , stddev = 2)
    with tf.Session() as session:
        plt.hist(norm.eval() , normed = True)
        plt.show()
        
def random_seed():
    uniform_with_seed = tf.random_uniform([1] , seed =1)
    uniform_without_seed = tf.random_uniform([1])

    print("First Run:")

    with tf.Session() as first_session:
        print("uniform with (seed = 1) = {}" .format(first_session.run(uniform_with_seed)))
        print("uniform with (seed = 1) = {}" .format(first_session.run(uniform_with_seed)))
        print("uniform with = {}" .format(first_session.run(uniform_without_seed)))
        print("uniform with = {}" .format(first_session.run(uniform_without_seed)))

    print("Second Run:")

    with tf.Session() as second_session:
        print("uniform with (seed = 1) = {}" .format(second_session.run(uniform_with_seed)))
        print("uniform with (seed = 1) = {}" .format(second_session.run(uniform_with_seed)))
        print("uniform with = {}" .format(second_session.run(uniform_without_seed)))
        print("uniform with = {}" .format(second_session.run(uniform_without_seed)))

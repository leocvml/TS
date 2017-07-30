
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

def display_partition(x_values , y_values , assignment_values):
    labels = []
    colors = ["red" , "blue" , "green" , "yellow"]
    for i in range(len(assignment_values)):
        labels.append(colors[(assignment_values[i])])
    color = labels
    df = pd.DataFrame(dict(x = x_values , y = y_values , color = labels))
    fig , ax = plt.subplots()
    ax.scatter(df['x'] , df['y'] , c=df['color'])
    plt.show()



num_vectors = 1000
num_clusters = 4
num_steps = 100

x_values = []
y_values = []
vector_values = []

for i in range(num_vectors):
    if np.random.random() > 0.5 :
        x_values.append(np.random.normal(0.4,0.7))
        y_values.append(np.random.normal(0.2,0.8))
    else :
        x_values.append(np.random.normal(0.6,0.4))
        y_values.append(np.random.normal(0.8,0.5))
       
vector_values = zip(x_values , y_values)
vector_values = list(vector_values)
vectors = tf.constant(vector_values)


n_sample = tf.shape(vector_values)[0]
random_indices = tf.random_shuffle(tf.range(0,n_sample))

begin = [0,]
size = [num_clusters,]
size[0] = num_clusters


centroid_indices = tf.slice(random_indices , begin , size)
centroids = tf.Variable(tf.gather(vector_values , centroid_indices))

expanded_vectors = tf.expand_dims(vectors , 0)
expanded_centroids = tf.expand_dims(centroids , 1)

vector_subtration = tf.subtract(expanded_vectors,expanded_centroids)

euclidean_distances = tf.reduce_sum(tf.square(vector_subtration) , 2 )
#assignments = tf.to_int32(tf.argmin(euclidean_distances) , 0)
assignments = tf.to_int32(tf.argmin(euclidean_distances)) 

partitions = tf.dynamic_partition(vectors, assignments, num_clusters)
update_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition , 0),0)for partition in partitions] , 0)
 






init_op =tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
for step in range(num_steps):
    _,centroid_values , assignment_values = sess.run([update_centroids,centroids,assignments])
display_partition(x_values , y_values , assignment_values)
plt.plot(x_values , y_values , 'o' , label = 'Input Data')
plt.legend()
plt.show()



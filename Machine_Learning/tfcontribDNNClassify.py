from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING ="C:/Users/GGmanxd/Desktop/python/Tensorflow/iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST ="C:/Users/GGmanxd/Desktop/python/Tensorflow/iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # If the training and test sets aren't stored locally, download them.
  print("OK")  
  if not os.path.exists(IRIS_TRAINING):
      raw = urllib.request.urlopen(IRIS_TRAINING_URL).read().decode()
      with open(IRIS_TRAINING, 'w') as f:
          f.write(raw)
          print(raw)
  if not os.path.exists(IRIS_TEST):
      raw = urllib.request.urlopen(IRIS_TEST_URL).read().decode()
      with open(IRIS_TEST, 'w') as f:
          f.write(raw)
          print(raw)          
  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)
  
  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                               hidden_units=[10,20,10,5],
                                               activation_fn=tf.nn.relu,
                                               dropout=0.2,
                                               n_classes=3,
                                               optimizer="Adam",
                                               model_dir="C:/Users/GGmanxd/Desktop/python/Tensorflow/ckpt")
  
  def epoch(epoch):
    for i in range(epoch):
      classifier.fit(input_fn=get_train_inputs, steps=1)
      accuracy_score = classifier.evaluate(input_fn=get_test_inputs,steps=1)["accuracy"]
      print("epoch:{i} Test Accuracy: {acc}\n".format(i = i,acc = accuracy_score))
    
  # Define the training inputs
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

  epoch(100)
  

  
    


  # Classify two new flower samples.
  def new_samples():
    return np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

  predictions = list(classifier.predict(input_fn=new_samples))

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predictions))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    main()

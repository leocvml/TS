import tensorflow as tf
import skimage.io as io
import numpy as np

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

tfrecords_filename = 'C:/Users/VISION-LAB/Desktop/python/DCGAN/data/img_align_celeba_tfrecords/train0.tfrecords'
tfrecords_filename1 = 'C:/Users/VISION-LAB/Desktop/python/DCGAN/data/img_align_celeba_tfrecords/train1.tfrecords'

def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([],tf.int64),
            'weight': tf.FixedLenFeature([],tf.int64),
            'depth' : tf.FixedLenFeature([],tf.int64),
            'image_raw': tf.FixedLenFeature([],tf.string)})



    image = tf.decode_raw(features['image_raw'], tf.uint8)

    height = tf.cast(features['height'],tf.int32)
    width = tf.cast(features['weight'],tf.int32)
    depth = tf.cast(features['depth'],tf.int32)

    image_shape = tf.stack([height,width,3])

    image = tf.reshape(image,image_shape)
    
    image_size_const = tf.constant((IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype=tf.int32)

    resize_image = tf.image.resize_image_with_crop_or_pad(image = image,
                                                          target_height =IMAGE_HEIGHT,
                                                          target_width = IMAGE_WIDTH)

   


    
    images = tf.train.shuffle_batch([resize_image],
                                    batch_size =64,
                                    capacity = 30,
                                    num_threads = 2,
                                    min_after_dequeue=10)
    
    return images



filename_queue = tf.train.string_input_producer([tfrecords_filename],num_epochs =2,shuffle=True)
# ["file0", "file1"] or [("file%d" % i) for i in range(2)]


image = read_and_decode(filename_queue)

init= tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())


with tf.Session() as sess:
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(100):
        img = sess.run([image])

        print(img[0].shape)

        print('current batch')


        io.imshow(img[0][1])
        io.show()

    coord.request_stop()
    coord.join(threads)


'''
def read_my_file_format(filename_queue):
  reader = tf.SomeReader()
  key, record_string = reader.read(filename_queue)
  example, label = tf.some_decoder(record_string)
  processed_example = some_processing(example)
  return processed_example, label

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label = read_my_file_format(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch
'''

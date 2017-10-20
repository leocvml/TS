import tensorflow as tf
import skimage.io as io
import numpy as np

reconstructed_images = []
tfrecords_filename = 'C:/Users/VISION-LAB/Desktop/python/DCGAN/data/img_align_celeba_tfrecords/train0.tfrecords'
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)


num = 0
for string_record in record_iterator:
    num +=1
    print(num)
    example = tf.train.Example()
    example.ParseFromString(string_record)


    height = int(example.features.feature['height']
                                .int64_list
                                .value[0])
    

    width = int(example.features.feature['weight']
                               .int64_list
                               .value[0])

    
    depth = int(example.features.feature['depth']
                               .int64_list

                               .value[0])
    
    
    img_string = (example.features.feature['image_raw']
                                 .bytes_list
                                 .value[0])


    img_1d = np.fromstring(img_string,dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height,width,depth))

    reconstructed_images.append(reconstructed_img)

print(len(reconstructed_images))

io.imshow(reconstructed_images[0])
io.show()

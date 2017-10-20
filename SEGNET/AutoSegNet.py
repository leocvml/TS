from skimage import data, io, filters
import numpy as np
import os.path
import tensorflow as tf
import random
from skimage.transform import resize
import matplotlib.pyplot as plt 



IMAGE_SIZE = 256
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE



def diff2img(pred , ground):

    diff = np.power(pred - ground ,2)
    diff = np.sqrt(diff)
    diff = np.sum(diff)
    return diff


def normalization(image):
    minVal = image.min()
    Img_Min = minVal
    maxVal = image.max()
    Img_Max = maxVal
    rangeVal = maxVal - minVal
    image = ((image - minVal) / rangeVal) * 1 

    return image 


def normal_mask(image):
    minVal = image.min()
    maxVal = image.max()
    rangeVal = maxVal - minVal
    image = ((image - minVal) / rangeVal) 

    return image 


def keep_norm(image):
    rangeVal = Img_Max - Img_Min
    image = ((image - Img_Min) / rangeVal) * 1 

    return image 
# input Data
# input Train Data / output Train mask

train_imgs =[]
train_imgName = [x for x in os.listdir('Train_Img') if x.endswith('.tif')]

for index in range(len(train_imgName)):
    imgName = ('Train_Img/'+ train_imgName[index])
    image = io.imread(imgName)
    image = resize(image,(256,256),mode='reflect')
    image = normalization(image)
    train_imgs.append(np.array(image))

train_imgs = np.array(train_imgs).reshape(len(train_imgName) , IMAGE_PIXELS)
#train_imgs = normalization(train_imgs)

print(train_imgs.shape)
print(train_imgs[0][:50])

'''
img = train_imgs[0].reshape(IMAGE_SIZE,IMAGE_SIZE)
io.imshow(train_imgs[0].reshape(IMAGE_SIZE,IMAGE_SIZE),cmap = 'gray')
io.show()
'''

train_masks = []
train_maskName = [x for x in os.listdir('Train_Mask') if x.endswith('.tif')]

for index in range(len(train_maskName)):
    maskName = ('Train_Mask/' + train_maskName[index])
    image = io.imread(maskName)
    image = resize(image,(256,256),mode='reflect')
    #image = normal_mask(image)
    train_masks.append(np.array(image))
train_masks = np.array(train_masks).reshape(len(train_maskName) , IMAGE_PIXELS)
train_masks = normal_mask(train_masks)
train_masks = train_masks 
print(train_masks.shape)
print(train_masks[0][:50])

'''
img = train_masks[0].reshape(IMAGE_SIZE,IMAGE_SIZE)
io.imshow(train_masks[0].reshape(IMAGE_SIZE,IMAGE_SIZE),cmap = 'gray')
io.show()
'''

test_imgs =[]
test_imgName = [x for x in os.listdir('Test_Img') if x.endswith('.tif')]

for index in range(len(test_imgName)):
    imgName = ('Test_Img/' + test_imgName[index])
    image = io.imread(imgName)
    image = resize(image,(256,256),mode='reflect')
    image = normal_mask(image)
    test_imgs.append(np.array(image))
test_imgs = np.array(test_imgs).reshape(len(test_imgName) , IMAGE_PIXELS)
#test_imgs = keep_norm(test_imgs)
#test_imgs = test_imgs 
print(test_imgs.shape)
print(test_imgs[0][:50])




test_masks =[]
test_maskName = [x for x in os.listdir('Test_Mask') if x.endswith('.tif')]

for index in range(len(test_maskName)):
    maskName = ('Test_Mask/' + test_maskName[index])
    image = io.imread(maskName)
    image = resize(image,(256,256),mode='reflect')
    #image = normal_mask(image)
    test_masks.append(np.array(image))
test_masks = np.array(test_masks).reshape(len(test_maskName) , IMAGE_PIXELS)
test_masks = normal_mask(test_masks)
test_masks = test_masks 
print(test_masks.shape)
print(test_masks[0][:50])



#network function
def conv2d(img,w,b):
    return tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1,1,1,1],padding='SAME'),b))

def max_pool(img):
    return tf.nn.max_pool(img , ksize=[1,2,2,1] , strides=[1,2,2,1], padding='SAME')

def deconv2d(x,w,output_shape):
    return tf.nn.conv2d_transpose(x,w,output_shape, strides=[1,1,1,1], padding = 'SAME')

def max_unpool_2x2(x,shape):
    inference = tf.image.resize_nearest_neighbor(x, tf.stack([shape[1],shape[2]]))
    return inference


# Stochastic_batch
def stochastic_batch(img,img_mask,batch_size):
    randIdx = np.random.randint(img.shape[0],size =batch_size)
    batchImg =[]
    batchImgmask = []
    for i in randIdx:
        batchImg.append(np.array(img[i]))
        batchImgmask.append(np.array(img_mask[i]))
    batchImg = np.array(batchImg)
    batchImgmask = np.array(batchImgmask)
    return batchImg , batchImgmask 


def batchData(Data,Target, batch_size , batch_num,index):
    img = []
    mask = []
    start = batch_size * batch_num
    end = min((batch_size * (batch_num+1)) , len(Data))
    for num in range(start,end):
        img.append(Data[num])
        mask.append(Target[num])

    img = np.array(img)
    mask = np.array(mask)

    return img , mask

#hyper parameter 
learning_rate = 0.001
training_epochs = 3000
batch_size = 100 #int(0.8 * len(train_imgName))
display_step = 1
examples_to_show = 10
#network arch

kernel_size = 9

def encode(img_data):

    # conv = 9x9 input = 1 output = 32
    wc1 = tf.Variable(tf.random_normal([kernel_size,kernel_size,1,32]))
    bc1 = tf.Variable(tf.random_normal([32]))

    # conv = 9x9 input = 32 output = 64
    wc2 = tf.Variable(tf.random_normal([3,3,32,64]))
    bc2 = tf.Variable(tf.random_normal([64]))

    flatten_data = tf.reshape(img_data,shape=[-1 ,IMAGE_SIZE ,IMAGE_SIZE ,1])
    conv1 = conv2d(flatten_data,wc1,bc1)
    pool_conv1 = max_pool(conv1)
    conv2 = conv2d(pool_conv1,wc2,bc2)
    pool_conv2 = max_pool(conv2)

    print("code layer shape : %s" % pool_conv2.get_shape())
    return pool_conv2


def decode (ori_img, code):

    #reconstruct 1
    w_dc1 = tf.Variable(tf.random_normal([kernel_size,kernel_size,32,64]))
    b_dc1 = tf.Variable(tf.random_normal([32]))
    
    output_shape_d_conv1 = tf.stack([tf.shape(ori_img)[0],64,64,32])
    d_conv1 = tf.nn.sigmoid(deconv2d(code,w_dc1,output_shape_d_conv1))
    
    output_shape_uppool1 = tf.stack([tf.shape(ori_img)[0], 128,128,32])
    up_d_conv1 = max_unpool_2x2(d_conv1,output_shape_uppool1)
    
    #reconstruct 2
    dconv3_size = IMAGE_SIZE 
    w_dc2 = tf.Variable(tf.random_normal([5,5,1,32]))
    b_dc2 = tf.Variable(tf.random_normal([1]))
    
    output_shape_d_conv2 = tf.stack([tf.shape(ori_img)[0], 128, 128, 1])
    h_d_conv2 = tf.nn.sigmoid(deconv2d(up_d_conv1,w_dc2,output_shape_d_conv2))

    output_shape_uppool2 = tf.stack([tf.shape(ori_img)[0],256,256,1])
    up_h_dconv2 = max_unpool_2x2(h_d_conv2,output_shape_uppool2)
    print("reconstruct layer shape : %s" % up_h_dconv2.get_shape())
    return up_h_dconv2
    


# Network Parameters
n_input = IMAGE_SIZE*IMAGE_SIZE
    
# tf graph input (only picture)

Img = tf.placeholder("float" , [None,n_input])

#######autoencoder 
_Img = tf.reshape(Img,shape = [-1,IMAGE_SIZE,IMAGE_SIZE,1])

#Model
encoder_op = encode(Img)
decoder_op = decode(Img,encoder_op)

'''
#Predict
y_pred = decoder_op
#######autoencoder 
y_true = _Img
'''


#Target Mask
y_pred = decoder_op
mask = tf.placeholder("float" , [None,n_input])
flatten_mask = tf.reshape(mask,shape=[-1,IMAGE_SIZE,IMAGE_SIZE,1])
y_true = flatten_mask




#define lost function
cost = tf.reduce_mean(tf.pow((y_pred - y_true), 2))

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

index = np.arange(len(train_imgs))

folder = "\\1005_1"


saver = tf.train.Saver(max_to_keep=None)
#Lauch the graph
for  train_num in range(5):
    kidfolder = '\\'+str(train_num)
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        total_batch = int(len(train_imgs) / batch_size) if (len(train_imgs) % batch_size) == 0 else int(len(train_imgs) / batch_size) + 1
        #print(total_batch)
        #Training cycle
        for epoch in range(training_epochs):
            np.random.shuffle(index)   
           #Loop over all batches
            for part_batch in range(total_batch):
                
                batch_Img,batchImgmask = batchData(train_imgs,train_masks,batch_size,part_batch,index)
            #print(batch_Img.shape)
            
                _ , c = sess.run([optimizer,cost], feed_dict ={Img:batch_Img , mask:batchImgmask })

            
            if epoch % display_step == 0:
                print("Epoch:" , "%04d" %(epoch+1), "cost = ","{:.9}".format(c))


        print("Optimization Finished")


        
        #batch_test = train_batch(train_imgs,batch_size)
        Img_test = test_imgs[:examples_to_show]
        Imgmask_test = test_masks[:examples_to_show]
    
        # Applying encode and decode over test set
        encode_decode = sess.run(
            y_pred, feed_dict={Img: Img_test[:examples_to_show] , mask:Imgmask_test[:examples_to_show]})

        # Compare original images with their reconstructions
        
        f, a = plt.subplots(3, 10, figsize=(examples_to_show+2, 5))
        encode_decode_reshape = []
        for i in range(examples_to_show):

            a[0][i].imshow(np.reshape(Img_test[i], (IMAGE_SIZE, IMAGE_SIZE)) ,cmap='gray' ,label = 'ya')
            a[1][i].imshow(np.reshape(encode_decode[i],(IMAGE_SIZE, IMAGE_SIZE)),cmap = 'gray')
            a[2][i].imshow(np.reshape(Imgmask_test[i],(IMAGE_SIZE, IMAGE_SIZE)),cmap = 'gray')
            

            encode_decode_reshape.append(np.reshape(encode_decode[i],(IMAGE_SIZE*IMAGE_SIZE)))
           # print(mask_reshape[i].shape)
           # print(Imgmask_test[i].shape)

            diff = diff2img(encode_decode_reshape[i] , Imgmask_test[i])
            diff = (1- (diff / IMAGE_PIXELS))
            a[0][i].set_title("{:.3}".format(diff))

        f.savefig("result_img/" +str(train_num) + ".png")
        #f.show()
        plt.close(f)
        
        path =r"C:\Users\VISION-LAB\Desktop\python\SEGNET\modelstore" + folder + kidfolder
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
            print("OK")

        modelName  =  path+"\\"  + str(train_num) +".ckpt"
        save_path = saver.save(sess, modelName)
        print("Model saved in file: %s" % save_path)






































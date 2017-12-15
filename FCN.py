from mxnet import image
import sys
sys.path.append('..')
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
from mxnet.gluon import nn
import matplotlib.pyplot as plt
from mxnet import image

from skimage import data, io, filters
import os.path


def read_images(train=True):
    txt_fname = 'VOCdevkit/VOC2012/ImageSets/Segmentation/' + (
        'train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    n = len(images)
    data, label = [None] * n, [None] * n
    for i, fname in enumerate(images):
        data[i] = image.imread('VOCdevkit/VOC2012/JPEGImages/%s.jpg' % (fname))
        label[i] = image.imread('VOCdevkit/VOC2012/SegmentationClass/%s.png' % (fname))
    return data, label


def rand_crop(data,label,height,width):
    data, rect =image.random_crop(data, (width,height))
    label = image.fixed_crop(label, *rect)
    return data,label



classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']
# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]




cm2lbl = np.zeros(256**3)

for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i
    

def image2label(im):
    data = im.astype('int32').asnumpy()
    idx = (data[:,:,0]*256 + data[:,:,1])*256+data[:,:,2]
    return nd.array(cm2lbl[idx])

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.299, 0.224, 0.225])

def normalize_image(data):
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std

class VOCSegDataset(gluon.data.Dataset):

    def _filter(self, images):
        return [im for im in images if (
            im.shape[0] >= self.crop_size[0] and
            im.shape[1] >= self.crop_size[1])]
    
    def __init__(self, train, crop_size):
        self.crop_size = crop_size
        data, label = read_images(train=train)
        data = self._filter(data)
        self.data = [normalize_image(im) for im in data]
        self.label = self._filter(label)
        print('Read ' + str(len(self.data)) + ' examples')

    def __getitem__(self,idx):
        data,label = rand_crop(
            self.data[idx], self.label[idx],
            *self.crop_size)
        data = data.transpose((2,0,1))
        label = image2label(label)
        return data,label

    def __len__(self):
        return len(self.data)
    

# height x width

input_shape = (320,480)
voc_train = VOCSegDataset(True,input_shape)
voc_test = VOCSegDataset(False,input_shape)




batch_size = 64
train_data = gluon.data.DataLoader(
    voc_train, batch_size=64, shuffle=True,last_batch='discard')
test_data = gluon.data.DataLoader(
    voc_test, batch_size=64,last_batch='discard')

for data,label in test_data:
    break
print(data.shape)
print(label.shape)


from mxnet.gluon.model_zoo import vision as models

pretrained_net = models.resnet18_v2(pretrained=True)

print(pretrained_net.features[-4:], pretrained_net.output)

net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)

x = nd.random.uniform(shape=(1,3,*input_shape))
print('Input:', x.shape)
print('Output:', net(x).shape)
    


'''
train_images, train_labels = read_images()
y = image2label(train_labels[1])
print(y[152:162,222:232])
'''


'''

train_images, train_labels = read_images()

print('read data OK!!')

imgs = []
for i in range(3):
    imgs += rand_crop(train_images[i], train_labels[i],200,300)


example_to_show = 3

f, a = plt.subplots(2, example_to_show, figsize=(example_to_show+2, 5))
for i in range(example_to_show):
    a[0][i].imshow(imgs[2*i].asnumpy())
    a[1][i].imshow(imgs[2*i+1].asnumpy())
f.show()
'''

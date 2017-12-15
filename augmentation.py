import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
from mxnet.gluon import nn
import matplotlib.pyplot as plt
from mxnet import image
from skimage import data, io, filters


def apply(img,aug,n=2):
    img2f = [aug(img.astype('float32')) for _ in range(n*n)]
    print(len(img2f))
    aug_img = nd.stack(*img2f).clip(0,255)/255

    f, a = plt.subplots(1, n*n, figsize=(8, 8))
    for i in range(n*n):
        a[i].imshow(aug_img[i].asnumpy())
    f.show()


mx.random.seed(1)
ctx = mx.gpu()

img = image.imdecode(open('CATFORAUG.png','rb').read())

augHor = image.HorizontalFlipAug(.5)   #水平翻轉

augCrop = image.RandomCropAug([200,200])  # random crop image to 200x200

augCropRand = image.RandomSizedCropAug((200,200),0.1,(.5,2))
#保留至少0.1區域  隨機長寬比在0.5~2之間 resize to 200x200

augBright = image.BrightnessJitterAug(.5)
#random bright increase or decrease in 50%

augHue = image.HueJitterAug(.5)


apply(img,augBright)


def get_transform(augs):
    def transform(data,label):
        #data: sample x height x width x channel
        #label: sample
        data = data.astype('float32')
        if augs is not None:
            data = nd.stack(*[apply_aug_list(d,augs) for d in data])
        data = nd.transpose(data,(0,3,1,2))
        return data, label.astype('float32')
    return transform



def get_data(batch_size, train_augs, test_augs =None):
    cifar10_train = gluon.data.vision.CIFAR10(
        train=True, transform=get_transform(train_augs))
    cifar10_test = gluon.data.vision.CIFAR10(
        train=False, transform=get_transform(test_augs))
    train_data = utils.DataLoader(
        cifar10_train, batch_size, shuffle=True)
    test_data = utils.DataLoader(
        cifar10_test, batch_size, shuffle=False)
    return (train_data, test_data)




    

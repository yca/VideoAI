#!/usr/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy

# Make sure that caffe is on the python path:
caffe_root = '/home/amenmd/myfs/tasks/cuda/caffe_master/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import matplotlib.cm as cm
import caffe

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, fileName, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data, cmap='jet')
    #plt.show()
    plt.savefig(fileName)
    #cv2.imwrite(fileName,data.astype(np.uint8))

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/amenmd/myfs/tasks/cuda/caffe_master/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '/home/amenmd/myfs/tasks/cuda/caffe_master/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

#MODEL_FILE = '/home/amenmd/myfs/tasks/cuda/caffe_master/caffe/models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers_deploy.prototxt'
#PRETRAINED = '/home/amenmd/myfs/tasks/cuda/caffe_master/caffe/models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers.caffemodel'
#ppmode = 1

caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
#transformer.set_mean('data', np.array([129.1863,104.7624,93.5940])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

input_image = caffe.io.load_image(sys.argv[1])
net.blobs['data'].reshape(1,3,227,227)
#net.blobs['data'].reshape(1,3,224,224)
net.blobs['data'].data[...] = transformer.preprocess('data', input_image)

out = net.forward()
#print("Predicted class is #{}.".format(out['prob'].argmax()))


#[(k, v.data.shape) for k, v in net.blobs.items()]
#plt.imshow(input_image)
#plt.show()

layer=sys.argv[2]
iid = "_" + sys.argv[3]

#filters = net.params['conv1'][0].data
#vis_square(filters.transpose(0, 2, 3, 1), layer + iid + '.jpg')

feat = net.blobs[layer].data[0]
vis_square(feat, 'feat' + iid + '.jpg', padval=1)

#filters = net.params['conv2'][0].data
#vis_square(filters[:48].reshape(48**2, 5, 5), 'conv2.png')
#feat = net.blobs['conv2'].data[0, :36]
#vis_square(feat, 'feat2.png', padval=1)


#feat = net.blobs['conv3'].data[0]
#vis_square(feat, 'conv3.png', padval=0.5)

#feat = net.blobs['conv4'].data[0]
#vis_square(feat, 'conv4.png', padval=0.5)

#feat = net.blobs['conv5'].data[0]
#vis_square(feat, 'conv5.png', padval=0.5)

#feat = net.blobs['pool5'].data[0]
#vis_square(feat, 'pool5.png', padval=1)

#!/usr/bin/env python
import os
import sys
import glob
import urllib
import tarfile
import numpy as np
from scipy.io import loadmat

def download_file(url, dest=None):
    if not dest:
        dest = 'data/' + url.split('/')[-1]
    urllib.urlretrieve(url, dest)
    
# Download the Oxford102 dataset into the current directory
if not os.path.exists('data'):
    os.mkdir('data')
    
    print("Downloading images...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz')
    tarfile.open("data/102flowers.tgz").extractall(path='data/')
    
    print("Downloading image labels...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat')
    
    print("Downloading train/test/valid splits...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat')
    
# Read .mat file containing training, testing, and validation sets.
setid = loadmat('data/setid.mat')

# The .mat file is 1-indexed, so we subtract one to match Caffe's convention.
idx_train = setid['trnid'][0] - 1
idx_test = setid['tstid'][0] - 1
idx_valid = setid['valid'][0] - 1

# Read .mat file containing image labels.
image_labels = loadmat('data/imagelabels.mat')['labels'][0]

# Subtract one to get 0-based labels
image_labels -= 1

files = sorted(glob.glob('data/jpg/*.jpg'))
labels = np.array(zip(files, image_labels))

# Get current working directory for making absolute paths to images
cwd = os.path.dirname(os.path.realpath(__file__))

def write_set_file(fout, labels):
    with open(fout, 'w+') as f:
        for label in labels:
            f.write('{0!s}/{1!s} {2!s}\n'.format(cwd, label[0], label[1]))

# Images are ordered by species, so shuffle them
np.random.seed(777)
idx_train = idx_train[np.random.permutation(len(idx_train))]
idx_test = idx_test[np.random.permutation(len(idx_test))]
idx_valid = idx_valid[np.random.permutation(len(idx_valid))]

write_set_file('train.txt', labels[idx_train,:])
write_set_file('test.txt', labels[idx_test,:])
write_set_file('valid.txt', labels[idx_valid,:])

if not os.path.exists('CaffeNet/pretrained-weights.caffemodel'):
    print('Downloading CaffeNet pretrained weights...')
    download_file('http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel', 
                  'CaffeNet/pretrained-weights.caffemodel')
    
if not os.path.exists('VGG_16/pretrained-weights.caffemodel'):
    print('Downloading VGG_16 pretrained weights...')
    download_file('http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel',
                  'VGG_16/pretrained-weights.caffemodel')
    
    
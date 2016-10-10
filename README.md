# Notice
Oxford102 is forked from [caffe-Oxford102](https://github.com/jimgoo/caffe-oxford102).
I modified some code and trained with VGG16 rather than VGG_S.I got better results than the original version

# caffe-oxford102

This bootstraps the training of deep convolutional neural networks with [Caffe](http://caffe.berkeleyvision.org/) to classify images in the [Oxford 102 category flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). A more detailed explanation can be found [here](http://jimgoo.com/flower-power/). The prototxt files for fine-tuning AlexNet and VGG_S models are included and use initial weights from training on the [ILSVRC 2012 (ImageNet) data](http://www.image-net.org/challenges/LSVRC/2012/). 

To download the Oxford 102 dataset, prepare Caffe image files, and download pre-trained model weights for CaffeNet and VGG_16, run

```bash
python bootstrap.py
```
This will give you some pretty flower pictures:

![alt tag](Oxford102/plots/flowers.png)

The categories are split into training, testing, and validation sets. It seems odd that there are more testing images than training images.

![alt tag](Oxford102/plots/splits.png)

## CaffeNet

Once you've run the `bootstrap.py` script, you can begin training from this directory with:

```bash
cd CaffeNet
caffe train -solver solver.prototxt -weights pretrained-weights.caffemodel -gpu 0
```


## VGG-16

To train,

```bash
cd VGG16
caffe train -solver solver.prototxt -weights pretrained-weights.caffemodel -gpu 0
```

## ResNet-50
If you want to use that:
you need running `convert_imageset.exe` script to get lmdb and downloading the model of resnet-50


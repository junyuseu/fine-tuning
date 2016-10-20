#!usr/bin/env python
'''
read a image of flowers and predict which kind of flowers it is.
'''
import caffe
import argparse
from class_labels import labels
import numpy as np
import matplotlib.pyplot as plt

#main function:parse the arument and predict
def main():
    parse=argparse.ArgumentParser()
    parse.add_argument(
        "input_file",
        help="Image file you want to predict"
    )
    parse.add_argument(
        "model",
        help="network structure"
    )
    parse.add_argument(
        "weights",
        help="pretrained model"
    )
    parse.add_argument(
        "mean_file",
        help="mena file"
    )
    parse.add_argument(
        "mean_size",
        type=int,
        help="test crop size of the original image.eg for CaffeNet is 227 and for VGGNet is 224"
    )
    args=parse.parse_args()
    image=caffe.io.load_image(args.input_file)
    if args.mean_size==224:
        imagenet_mean = np.load(args.mean_file)[:, 16:16 + 224, 16:16 + 224]
    elif args.mean_size==227:
        imagenet_mean = np.load(args.mean_file)[:, 14:14 + 227, 14:14 + 227]
    net=caffe.Classifier(
        args.model,args.weights,
        mean=imagenet_mean,# subtract the dataset-mean value in each channel
        channel_swap=(2,1,0),# swap channels from RGB to BGR
        raw_scale=255,# rescale from [0, 1] to [0, 255]
        image_dims=(256,256)
    )
    result=net.predict([image])
    label=np.argmax(result)
    plt.imshow(image)
    plt.axis('off')
    plt.title('{}:{:.3f}'.format(labels[label],result[0][label]))
    plt.savefig('{}.png'.format(labels[label]))
    plt.show()

if __name__=='__main__':
    main()
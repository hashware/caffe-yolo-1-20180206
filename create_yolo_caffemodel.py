# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:10:21 2016

@author: xingw, Banus
"""
from __future__ import print_function

import argparse

import numpy as np
import caffe


def transpose_matrix(array, rows, cols):
    """ transpose flattened matrix """
    return array.reshape((rows, cols)).transpose().flatten()


def load_parameter(weights, layer_data, transpose=False):
    """  load Caffe parameters from YOLO weights """
    shape = layer_data.shape
    size = np.prod(shape)
    if transpose:
        layer_data[...] = np.reshape(
            transpose_matrix(weights[:size], shape[1], shape[0]), shape)
    else:
        layer_data[...] = np.reshape(weights[:size], shape)

    return size


def convert_weights(model_filename, yoloweight_filename, caffemodel_filename):
    """ convert YOLO weights to .caffemodel format given the caffe model """
    net = caffe.Net(model_filename, caffe.TEST)

    # read header to get the transpose flag
    weights_int = np.fromfile(yoloweight_filename, dtype=np.int32, count=4)
    # transpose flag, the first 4 entries are major, minor, revision and net.seen
    transp_flag = (weights_int[0] > 1000 or weights_int[1] > 1000)
    print('Transpose fc layers: {}'.format(transp_flag))

    # read the weights from YOLO file, skipping the header
    weights = np.fromfile(yoloweight_filename, dtype=np.float32)[4:]

    count = 0
    for name, layer in zip(net.top_names, net.layers):
        if name not in net.params.keys():  # layer without parameters
            continue

        print("  converting {0}".format(name))
        count += load_parameter(weights[count:], net.params[name][1].data) # bias

        if   layer.type == 'Convolution':
            bn_name = "{0}_BN".format(name)
            if bn_name in net.top_names:  # there is a batchnorm layer
                # scales, rolling mean, rolling variance
                count += load_parameter(weights[count:], net.params[bn_name][0].data)
                count += load_parameter(weights[count:], net.params[bn_name][1].data)
            # weights
            count += load_parameter(weights[count:], net.params[name][0].data, transp_flag)
        elif layer.type == 'InnerProduct':   # fc layer
            count += load_parameter(weights[count:], net.params[name][0].data, transp_flag)
        elif layer.type == 'BatchNorm':
            continue   # handled within the convolutional layer
        else:
            print("WARNING: unknown type {} for {}".format(layer.type, name))

    if count != weights.shape[0]:  # weights left out
        raise ValueError(" Wrong number of weights: read {0}, used {1} (missing {2})".
                         format(weights.shape[0], count, weights.shape[0]-count))
    print('Converted {0} weights.'.format(count))
    net.save(caffemodel_filename)


def main():
    """ script entry point """
    parser = argparse.ArgumentParser(description='Convert YOLO weights to Caffe.')
    parser.add_argument('model', type=str, help='Caffe model file')
    parser.add_argument('yolo_weights', type=str, help='YOLO weight file')
    parser.add_argument('output', type=str, help='converted .caffemodel')
    args = parser.parse_args()

    print('model file is {}'.format(args.model))
    print('weight file is {}'.format(args.yolo_weights))
    print('output caffemodel file is {}'.format(args.output))

    convert_weights(args.model, args.yolo_weights, args.output)


if __name__ == '__main__':
    main()

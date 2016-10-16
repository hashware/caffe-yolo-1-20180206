# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:10:21 2016

@author: xingw
"""
from __future__ import print_function

import getopt
import sys

import caffe
import numpy as np


def transpose_matrix(array, rows, cols):
    """ transpose flattened matrix """
    return array.reshape((rows, cols)).transpose().flatten()


def convert_weights(model_filename, yoloweight_filename, caffemodel_filename):
    """ convert YOLO weights to .caffemodel format given the caffe model """
    net = caffe.Net(model_filename, caffe.TEST)
    layers = net.params.keys()

    # read header to get the transpose flag
    weights_int = np.fromfile(yoloweight_filename, dtype=np.int32, count=4)
    # transpose flag, the first 4 entries are major, minor, revision and net.seen
    transp_flag = (weights_int[0] > 1000 or weights_int[1] > 1000)
    print('Transpose FC: {}'.format(transp_flag))

    # read the weights from YOLO file, skipping the header
    weights = np.fromfile(yoloweight_filename, dtype=np.float32)[4:]
    print('Weigts to convert: {}'.format(weights.shape[0]))

    count = 0
    for layer in layers:
        bias_size = np.prod(net.params[layer][1].data.shape)
        net.params[layer][1].data[...] = \
            np.reshape(weights[count:count+bias_size], net.params[layer][1].data.shape)
        count += bias_size
        weight_size = np.prod(net.params[layer][0].data.shape)
        if layer[0:2] == 'co': # convolutional layer
            net.params[layer][0].data[...] = \
                np.reshape(weights[count:count+weight_size], net.params[layer][0].data.shape)
        else: # fc layer
            dims = net.params[layer][0].data.shape
            if transp_flag: # need to transpose for fc layers
                net.params[layer][0].data[...] = \
                    np.reshape(transpose_matrix(
                        weights[count:count+weight_size], dims[1], dims[0]), dims)
            else:
                net.params[layer][0].data[...] = np.reshape(weights[count:count+weight_size], dims)
        count += weight_size

    print('Converted {} weights.'.format(count))
    if count != weights.shape[0]:  # weights left out
        raise ValueError(" Wring number of weights: read {}, used {}".
                         format(weights.shape[0], count))
    net.save(caffemodel_filename)


def main(argv):
    """ script entry point """
    model_filename = ''
    yoloweight_filename = ''
    caffemodel_filename = ''
    usage_message = 'Usage: create_yolo_caffemodel.py -m <caffe_model_file> '\
                    '-w <yoloweight_filename> -o <caffemodel_output>'
    try:
        opts, _ = getopt.getopt(argv, "hm:w:o:")
        print('Options: {}'.format(opts))
    except getopt.GetoptError:
        print(usage_message)
        return
    if not opts:
        print(usage_message)
        return

    for opt, arg in opts:
        if opt == '-h':
            print(usage_message)
            return
        elif opt == "-m":
            model_filename = arg
        elif opt == "-w":
            yoloweight_filename = arg
        elif opt == "-o":
            caffemodel_filename = arg

    print('model file is {}'.format(model_filename))
    print('weight file is {}'.format(yoloweight_filename))
    print('output caffemodel file is {}'.format(caffemodel_filename))

    convert_weights(model_filename, yoloweight_filename, caffemodel_filename)


if __name__ == '__main__':
    main(sys.argv[1:])

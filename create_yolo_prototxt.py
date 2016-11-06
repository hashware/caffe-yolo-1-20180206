# -*- coding: utf-8 -*-
"""
Convert a YOLO's .cfg to Caffe's .prototxt
"""
from __future__ import print_function

import argparse

from caffe import layers as cl
from caffe import params as cp
import caffe


def load_configuration(fname):
    """ Load YOLO configuration file. """
    with open(fname, 'r') as fconf:
        lines = [l.strip() for l in fconf]

    config = []
    element = {}
    section_name = None
    for line in lines:
        if not line:
            continue
        if line[0] == '[':  # new section
            if section_name:
                config.append((section_name, element))
                element = {}
            section_name = line[1:].strip(']')
        else:
            key, value = line.split('=')
            element[key] = value
    config.append((section_name, element))

    return config


## Layer parsing ##
###################


def data_layer(name, params, deploy=True):
    """ add a data layer """
    fields = dict(shape={"dim": [1, int(params["channels"]),
                                 int(params["width"]), int(params["height"])]})
    if deploy:
        layer = cl.Input
    else:
        fields.update(data_param=dict(batch_size=int(params["batch"])),
                      include=dict(phase=caffe.TEST))
        layer = cl.DummyData
    return layer(name=name, **fields)


def convolutional_layer(previous, name, params, deploy=True):
    """ create a convolutional layer given the parameters and previous layer """
    fields = dict(num_output=int(params["filters"]),
                  kernel_size=int(params["size"]),
                  stride=int(params["stride"]))
    if not deploy:
        fields.update(weight_filler=dict(type="gaussian", std=0.01),
                      bias_filler=dict(type="constant", value=0))
    return cl.Convolution(previous, name=name, **fields)


def pooling_layer(previous, name, params):
    """ create a max pooling layer """
    return cl.Pooling(
        previous, name=name, pool=cp.Pooling.MAX,
        kernel_size=int(params["size"]), stride=int(params["stride"]))


def dense_layer(previous, name, params, deploy=True):
    """ create a densse layer """
    fields = dict(num_output=int(params["output"]))
    if not deploy:
        fields.update(weight_filler=dict(type="gaussian", std=0.01),
                      bias_filler=dict(type="constant", value=0))
    return cl.InnerProduct(previous, name=name, inner_product_param=fields)


def convert_configuration(config, deploy=True):
    """ given a list of YOLO layers as dictionaries, convert them to Caffe """
    model = caffe.NetSpec()
    count = 0

    for section, params in config:
        if   section == "net":
            last_layer = data_layer("data", params, deploy)
            setattr(model, "data", last_layer)
        elif section == "convolutional":
            count += 1
            layer_name = "conv{}".format(count)
            if params["batch_normalize"] == '1':
                bn_name = layer_name + "_BN"
                last_layer = cl.BatchNorm(last_layer, name=bn_name)
                setattr(model, bn_name, last_layer)
            last_layer = convolutional_layer(last_layer, layer_name, params, deploy)
            setattr(model, layer_name, last_layer)
            if params["activation"] == "leaky":
                layer_name = "relu{}".format(count)
                last_layer = cl.ReLU(last_layer, name=layer_name,
                                     in_place=True, relu_param=dict(negative_slope=0.1))
                setattr(model, layer_name, last_layer)
        elif section == "maxpool":
            layer_name = "pool{}".format(count)
            last_layer = pooling_layer(last_layer, layer_name, params)
            setattr(model, layer_name, last_layer)
        elif section == "connected":
            count += 1
            layer_name = "fc{}".format(count)
            last_layer = dense_layer(last_layer, layer_name, params, deploy)
        else:
            print("WARNING: {0} layer not recognized".format(section))

    model.out = last_layer

    return model


def main():
    """ script entry point """
    parser = argparse.ArgumentParser(description='Convert a YOLO cfg file.')
    parser.add_argument('model', type=str, help='YOLO cfg model')
    parser.add_argument('output', type=str, help='output prototxt')
    parser.add_argument('--deploy', action='store_true',
                        help='generate deploy prototxt')
    args = parser.parse_args()

    config = load_configuration(args.model)
    model = convert_configuration(config, args.deploy)

    suffix = "deploy" if args.deploy else "train_val"

    with open("{}_{}.prototxt".format(args.output, suffix), 'w') as fproto:
        fproto.write('%s\n' % model.to_proto())


if __name__ == '__main__':
    main()

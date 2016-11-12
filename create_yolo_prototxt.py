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
        if not line or line[0] == '#':  # empty or comment
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


def data_layer(name, params, train=False):
    """ add a data layer """
    fields = dict(shape={"dim": [1, int(params["channels"]),
                                 int(params["width"]), int(params["height"])]})
    if train:
        fields.update(data_param=dict(batch_size=int(params["batch"])),
                      include=dict(phase=caffe.TEST))
        if "crop_width" in params.keys():
            if params["crop_width"] != params["crop_height"]:
                raise ValueError("Rectangular crop not supported.")
            fields.update(transform_param=dict(
                mirror=bool(params["flip"]), crop_size=int(params["crop_width"])))
        layer = cl.DummyData
    else:
        layer = cl.Input

    return layer(name=name, **fields)


def activation_layer(previous, count, mode="relu"):
    """ create a non-linear activation layer """
    if   mode == "relu":
        return cl.RelU(previous, name="relu{}".format(count), in_place=True)
    elif mode == "leaky":
        return cl.ReLU(previous, name="relu{}".format(count),
                       in_place=True, relu_param=dict(negative_slope=0.1))
    else:
        raise ValueError("Activation mode not implemented: {0}".format(mode))


def convolutional_layer(previous, name, params, train=False):
    """ create a convolutional layer given the parameters and previous layer """
    fields = dict(num_output=int(params["filters"]),
                  kernel_size=int(params["size"]))
    if "stride" in params.keys():
        fields["stride"] = int(params["stride"])

    if int(params.get("pad", 0)) == 1:    # use 'same' strategy for convolutions
        fields["pad"] = fields["kernel_size"]//2
    if train:
        fields.update(weight_filler=dict(type="gaussian", std=0.01),
                      bias_filler=dict(type="constant", value=0))

    return cl.Convolution(previous, name=name, **fields)


def max_pooling_layer(previous, name, params):
    """ create a max pooling layer """
    return cl.Pooling(
        previous, name=name, pool=cp.Pooling.MAX,
        kernel_size=int(params["size"]), stride=int(params["stride"]))


def dense_layer(previous, name, params, train=False):
    """ create a densse layer """
    fields = dict(num_output=int(params["output"]))
    if train:
        fields.update(weight_filler=dict(type="gaussian", std=0.01),
                      bias_filler=dict(type="constant", value=0))
    return cl.InnerProduct(previous, name=name, inner_product_param=fields)


### layer aggregation ###
#########################

def add_convolutional_layer(layers, count, params, train=False):
    """ add layers related to a convolutional block in YOLO the layer list """
    layer_name = "conv{0}".format(count)
    layers.append(convolutional_layer(layers[-1], layer_name, params, train))
    if params.get("batch_normalize", 0) == '1':
        scale_name = "{0}_scale".format(layer_name)
        layers.append(cl.Scale(layers[-1], name=scale_name,
                               scale_param=dict(bias_term=True, axis=1)))
    if params["activation"] != "linear":
        layers.append(activation_layer(layers[-1], count, params["activation"]))


def add_dense_layer(layers, count, params, train=False):
    """ add layers related to a connected block in YOLO to the layer list """
    layer_name = "fc{0}".format(count)
    layers.append(dense_layer(layers[-1], layer_name, params, train))
    if params["activation"] != "linear":
        layers.append(activation_layer(layers[-1], count, params["activation"]))


def convert_configuration(config, train=False):
    """ given a list of YOLO layers as dictionaries, convert them to Caffe """
    layers = []
    count = 0

    for section, params in config:
        if   section == "net":
            input_params = params
            layers.append(data_layer("data", input_params, train))
        elif section == "crop":
            if train:    # update data layer
                input_params.update(params)
                layers[-1] = data_layer("data", input_params, train)
        elif section == "convolutional":
            count += 1
            add_convolutional_layer(layers, count, params, train)
        elif section == "maxpool":
            layers.append(max_pooling_layer(layers[-1], "pool{0}".format(count),
                                            params))
        elif section == "connected":
            count += 1
            add_dense_layer(layers, count, params, train)
        elif section == "dropout":
            if train:
                layers.append(cl.Dropout(layers[-1], name="drop{0}".format(count),
                                         dropout_ratio=float(params["probability"])))
        elif section == "local":  # locally connected layer
            count += 1
            raise ValueError("NOT IMPLEMENTED: {0}".format("local{}".format(count)))
        else:
            print("WARNING: {0} layer not recognized".format(section))

    model = caffe.NetSpec()
    for layer in layers:
        setattr(model, layer.fn.params["name"], layer)
    model.result = layers[-1]

    return model


def main():
    """ script entry point """
    parser = argparse.ArgumentParser(description='Convert a YOLO cfg file.')
    parser.add_argument('model', type=str, help='YOLO cfg model')
    parser.add_argument('output', type=str, help='output prototxt')
    parser.add_argument('--train', action='store_true',
                        help='generate train_val prototxt')
    args = parser.parse_args()

    config = load_configuration(args.model)
    model = convert_configuration(config, args.train)

    suffix = "train_val" if args.train else "deploy"

    with open("{}_{}.prototxt".format(args.output, suffix), 'w') as fproto:
        fproto.write('%s\n' % model.to_proto())


if __name__ == '__main__':
    main()

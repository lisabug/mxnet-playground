#!/usr/bin/env python
# -*- coding:utf-8 -*-

import mxnet as mx
import layer_normalization


def get_mlp(num_label, use_ln=False):
    data = mx.symbol.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=1000)
    if use_ln:
        fc1 = mx.sym.Custom(data=fc1, name='ln_fc1', op_type='layernormalization')
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 1000)
    if use_ln:
        fc2 = mx.sym.Custom(data=fc2, name='ln_fc2', op_type='layernormalization')
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_label)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

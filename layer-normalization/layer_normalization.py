#!/usr/bin/env python
# -*- coding:utf-8 -*-

import mxnet as mx


class LayerNormalization(mx.operator.CustomOp):
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon

    def forward(self, is_train, req, in_data, out_data, aux):
        x, gain, bias = in_data[0], in_data[1], in_data[2]
        assert len(x.shape) == 2, x.shape
        mu, sigma = self._statistcs(x)
        y = (x - mu) / sigma * gain + bias
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x, gain, bias = in_data[0], in_data[1], in_data[2]
        gain = mx.nd.expand_dims(gain, axis=0)  # 1xd for broadcast
        bias = mx.nd.expand_dims(bias, axis=0)
        mu, sigma = self._statistcs(x)

        dy = out_grad[0]
        dx = gain / sigma * dy
        dgain = mx.nd.sum((x - mu) / sigma * dy, axis=0)
        dbias = mx.nd.sum(dy, axis=0)
        self.assign(in_grad[0], req[0], dx)
        self.assign(in_grad[1], req[1], dgain)
        self.assign(in_grad[2], req[2], dbias)

    def _statistcs(self, x):
        dim = int(x.shape[1])
        mu_ = mx.nd.sum(x, axis=1) / dim
        mu = mx.nd.expand_dims(mu_, axis=1)
        sigma_ = mx.nd.sqrt(mx.nd.sum((x - mu) ** 2, axis=1) / dim)
        sigma = mx.nd.expand_dims(sigma_, axis=1)
        return mu, sigma


@mx.operator.register("layernormalization")
class LayerNormalizationProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(LayerNormalizationProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data', 'gamma', 'beta']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        gamma_shape = (in_shape[0][1],)
        beta_shape = (in_shape[0][1],)
        output_shape = in_shape[0]
        return [data_shape, gamma_shape, beta_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return LayerNormalization()

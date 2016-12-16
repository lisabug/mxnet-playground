#!/usr/bin/env python
# -*- coding:utf-8 -*-

import mxnet as mx
import numpy as np
import os
import struct


class MNISTIter(mx.io.DataIter):
    def __init__(self, image_path, label_path, batch_size,
            shuffle=True, rnn_mode=None, num_hidden=100):
        super(MNISTIter, self).__init__()
        self._label, self._data = self._read_data(label_path, image_path)
        self.data_shape = (batch_size, 1, 28, 28)
        self.label_shape = (batch_size, )
        self.batch_size = batch_size
        self.order = np.arange(len(self._data))
        self.num_data = self._data.shape[0]
        self.shuffle = shuffle
        self.cursor = -1

        self.data_buffer = mx.nd.zeros((self.data_shape), dtype=np.float32)
        self.label_buffer = mx.nd.zeros((self.label_shape), dtype=np.float32)

        self.rnn_mode = rnn_mode
        if rnn_mode:
            if rnn_mode == 'lstm':
                self._init_h = mx.nd.zeros((batch_size, num_hidden), dtype=np.float32)
                self._init_c = mx.nd.zeros((batch_size, num_hidden), dtype=np.float32)
                self._init_h_name = 'init_h'
                self._init_c_name = 'init_c'
            else:
                raise NotImplementError
        if shuffle:
            self._shuffle()

    def _shuffle(self):
        np.random.shuffle(self.order)

    def _read_data(self, label_path, image_path):
        with open(label_path) as f:
            magic, num = struct.unpack('>II', f.read(8))
            label = np.fromstring(f.read(), dtype=np.int8)
        with open(image_path) as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            image = np.fromstring(f.read(), dtype=np.uint8).reshape(len(label), rows, cols)
            image = self._to4d(image)
        return (label, image)

    def _to4d(self, image):
        return image.reshape(image.shape[0], 1, 28, 28).astype(np.float32)/255

    def reset(self):
        self.cursor = -self.batch_size
        if self.shuffle:
            self._shuffle()

    @property
    def provide_data(self):
        if not self.rnn_mode:
            return [('data', self.data_shape)]
        else:
            if self.rnn_mode == 'lstm':
                return [('data', self.data_shape),
                        (self._init_h_name, self._init_h.shape),
                        (self._init_c_name, self._init_c.shape)]
            else:
                raise NotImplementError

    @property
    def provide_label(self):
        return [('softmax_label', self.label_shape)]

    def _get_pad(self):
        if self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.num_data

    def next(self):
        if self.iter_next():
            pad = self._get_pad()
            if pad:
                data = np.concatenate((self._data[self.order[:pad]], self._data[self.order[self.cursor:self.num_data]]), axis=0)
                label = np.concatenate((self._label[self.order[:pad]], self._label[self.order[self.cursor:self.num_data]]), axis=0)
            else:
                data = self._data[self.order[self.cursor:self.cursor+self.batch_size]]
                label = self._label[self.order[self.cursor:self.cursor+self.batch_size]]
            self.data_buffer[:] = data
            self.label_buffer[:] = label
            return mx.io.DataBatch(data=[self.data_buffer, self._init_h, self._init_c] if self.rnn_mode else [self.data_buffer],
                                   label=[self.label_buffer],
                                   pad=pad,
                                   provide_data=self.provide_data,
                                   provide_label=self.provide_label,
                                   index=None)
        else:
            raise StopIteration

#!/usr/bin/env python
# -*- coding:utf-8 -*-

import mxnet as mx
import sys
sys.path.insert(0, "../common")
from iterator.mnist import MNISTIter
from symbol import get_mlp
import argparse
import logging


def main(args):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.debug(str(args))
    train_iter = MNISTIter("../data/mnist/train-images-idx3-ubyte",
                           "../data/mnist/train-labels-idx1-ubyte",
                           args.batch_size,
                           num_hidden=args.num_hidden)
    val_iter = MNISTIter("../data/mnist/t10k-images-idx3-ubyte",
                         "../data/mnist/t10k-labels-idx1-ubyte",
                         args.batch_size,
                         num_hidden=args.num_hidden,
                         shuffle=False)

    mlp = get_mlp(10, args.ln)

    ctx = mx.gpu(args.gpu) if args.gpu >=0 else mx.cpu(0)
    initializer = mx.initializer.Uniform(scale=0.1)
    optimizer = 'adam'
    optimizer_params = {'learning_rate': args.lr,
                        'wd': args.wd}
    metric = mx.metric.CompositeEvalMetric()
    metric.add(mx.metric.CrossEntropy())
    metric.add(mx.metric.Accuracy())

    mod = mx.mod.Module(mlp, context=ctx)
    mod.fit(train_iter, eval_data=val_iter, num_epoch=args.num_epoch,
            eval_metric=metric,
            batch_end_callback=mx.callback.Speedometer(args.batch_size, 50),
            initializer=initializer,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', dest='batch_size', type=int, default=100,
                        help='batch size, default is 100')
    parser.add_argument('--hidden', dest='num_hidden', type=int, default=64,
                        help='number of hidden state unit, default is 64')
    parser.add_argument('--gpu', type=int, default=0,
                        help='device of gpu to use, -1 means cpu')
    parser.add_argument('--epoch', dest='num_epoch', type=int, default=50,
                        help='number of epoch to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate, default is 1e-4')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='weight decay, default is 1e-5')
    parser.add_argument('--ln', action='store_true', help='use layer normalization')
    args = parser.parse_args()
    main(args)

__docformat__ = 'restructedtext en'

import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),dtype=theano.config.floatX),name='W',borrow=True)
        self.b = theano.shared(value=numpy.zeros((1,n_out),dtype=theano.config.floatX),name='b',borrow=True)

        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x,0)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def zeroOneLoss(self, y):
        return T.ceil((T.argmax(self.p_y_given_x)-T.argmax(y))/48.)

__author__ = 'jefftsai'

import os
import sys
import time
import numpy as np

import theano
import theano.tensor as T


from logistic_sgd import LogisticRegression

class hiddenLayer:
    def __init__(self,rng,input,nIn,nOut,W=None,b=None,activation=T.tanh):
        self.input = input
        # W 39x200
        if W is None:
            WValue = np.array(rng.uniform(low=-np.sqrt(6./(nIn+nOut)),high=np.sqrt(6./(nIn+nOut)),size=(nIn,nOut)),dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                WValue *= 4
            W = theano.shared(value=WValue,name="W",borrow=True)
        # b 200x1
        if b is None:
            bValue = np.zeros((1,nOut),dtype=theano.config.floatX)
            b = theano.shared(value=bValue,name="b",borrow=True)

        self.W = W
        self.b = b

        linOutput = T.dot(input,self.W)+self.b
        self.output = activation(linOutput)

        self.parameters = [self.W,self.b]

class mlp:
    def __init__(self,rng,input,nIn,nHidden,nOut):
        self.hiddenLayer = hiddenLayer(rng,input = input,nIn = nIn,nOut = nHidden,activation = T.tanh)
        self.hiddenLayer2 = hiddenLayer(rng,input = self.hiddenLayer.output,nIn = nHidden,nOut = nHidden,activation = T.tanh)
        self.hiddenLayer3 = hiddenLayer(rng,input = self.hiddenLayer2.output,nIn = nHidden,nOut = nHidden,activation = T.tanh)
        self.logisticRegressionLayer = LogisticRegression(input = self.hiddenLayer3.output,n_in=nHidden,n_out=nOut)

        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.hiddenLayer2.W).sum() + abs(self.hiddenLayer3.W).sum() + abs(self.logisticRegressionLayer.W).sum()
        self.L2 = (self.hiddenLayer.W**2).sum() + (self.hiddenLayer2.W**2).sum()+ (self.hiddenLayer3.W**2).sum() + (self.logisticRegressionLayer.W**2).sum()

        # self.cost = self.logisticRegressionLayer.negative_log_likelihood
        self.cost = self.logisticRegressionLayer.zeroOneLoss
        # self.output = self.logisticRegressionLayer.output
        self.parameters = self.hiddenLayer.parameters+self.hiddenLayer2.parameters+self.hiddenLayer3.parameters+self.logisticRegressionLayer.params

    def costFunction(self,y):
        return -T.sum((self.logisticRegressionLayer.y_pred-y)**2)

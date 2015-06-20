__author__ = 'root'

import numpy as np
import theano
import theano.tensor as T


class RNN:
    def __init__(self,x,y,layerNumber,memory=None,parameters=None):
        if parameters is None:
            self.inputLayer = hiddenLayer(x,layerNumber[0],layerNumber[1],memory)
            self.outputLayer = outputLayer(self.inputLayer.z,layerNumber[1],layerNumber[2])

        self.parameters = self.inputLayer.W+self.inputLayer.WH+self.outputLayer.WOut




class outputLayer:
    def __init__(self,x,inLayerNum,outLayerNum,W=None,activationFunction = T.nnet.softmax):
        lower = -0.01
        upper = 0.01

        # for output layer
        if W is None:
            WOutInitail = np.asarray(np.random.normal(size = (inLayerNum,outLayerNum),low = lower,high = upper),dtype = theano.config.floatX)
            self.WOut= theano.shared(value = WOutInitail,name = "WOut")

        self.activationFunction = activationFunction

        a = T.dot(x,self.WOut)
        y = self.activationFunction(a)

        self.yPred = y
        self.parameters = [self.WOut]

    def costFunction(self,y):
        return T.mean(T.nnet.categorical_crossentropy(self.yPred,y))


class hiddenLayer:
    def __init__(self,x,inLayerNum,outLayerNum,memory=None,W=None,WH=None,activationFunction = T.nnet.sigmoid):
        lower = -0.01
        upper = 0.01
        # for recurrent weight
        if WH is None:
            WHInitial = np.asarray(np.random.normal(size = (outLayerNum,outLayerNum),low = lower,high = upper),dtype = theano.config.floatX)
            self.WH = theano.shared(value = WHInitial,name = "WH")

        if W is None:
            WInitial = np.asarray(np.random.normal(size = (inLayerNum,outLayerNum),low = -0.01,high = 0.01),dtype = theano.config.floatX)
            self.W = theano.shared(value = WInitial,name = "W")

        if memory is None:
            memoryInitial = np.zeros((inLayerNum,),dtype = theano.config.floatX)
            self.memory = theano.shared(value=memoryInitial,name="memory")

        self.activationFunction = activationFunction

        a = T.dot(x,self.W)+T.dot(self.memory,self.WH)
        self.memory = a

        z = self.activationFunction(a)

        self.z = z
        self.parameters = [W,WH]
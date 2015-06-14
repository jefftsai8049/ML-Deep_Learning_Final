__author__ = 'root'

import numpy as np
import theano.tensor as T
import theano

class mlp:
    def __init__(self,x,y,layerNumber):
        randomSeed = np.random.RandomState(8888)

        self.inputLayer = hiddenLayer(randomSeed,x,layerNumber[0],layerNumber[1])
        self.hiddenLayer = hiddenLayer(randomSeed,self.inputLayer.z,layerNumber[1],layerNumber[2])
        self.hiddenLayer2 = hiddenLayer(randomSeed,self.hiddenLayer.z,layerNumber[2],layerNumber[3])
        self.outputLayer = outputLayer(randomSeed,self.hiddenLayer2.z,layerNumber[3],layerNumber[4])

        self.costFunction = outputLayer.costFunction
        self.parameters = self.inputLayer.parameters+self.hiddenLayer.parameters+self.hiddenLayer2.parameters+self.outputLayer.parameters


class hiddenLayer:
    def __init__(self,randomSeed,x,inLayer,outLayer,W=None,b=None,activationFunction = T.nnet.sigmoid):
        lower = -0.01
        upper = 0.01
        if W is None:
            WValue = np.array(randomSeed.uniform(low=lower,high=upper,size=(inLayer,outLayer)),dtype=theano.config.floatX)
            W = theano.shared(value=WValue,name="W",borrow=True)
            # W = W.get_value()

        if b is None:
            bValue = np.zeros((1,outLayer),dtype=theano.config.floatX)
            b= theano.shared(value=bValue,name="b",borrow=True)
        self.W = W
        self.b = b
        a = T.dot(x,self.W)+self.b
        z = activationFunction(a)
        self.z = z
        self.parameters = [self.W,self.b]

class outputLayer:
    def __init__(self,randomSeed,x,inLayer,outLayer,W=None,b=None,activationFunction = T.nnet.sigmoid):
        lower = -0.01
        upper = 0.01
        if W is None:
            WValue = np.array(randomSeed.uniform(low=lower,high=upper,size=(inLayer,outLayer)),dtype=theano.config.floatX)
            W = theano.shared(value=WValue,name="W",borrow=True)
            # W = W.get_value()

        if b is None:
            bValue = np.zeros((1,outLayer),dtype=theano.config.floatX)
            b= theano.shared(value=bValue,name="b",borrow=True)

        self.W = W
        self.b = b

        a = T.dot(x,self.W)+self.b
        y = activationFunction(a)
        self.parameters = [self.W,self.b]
        self.yPred = y

    def costFunction(self,y):
        return T.nnet.categorical_crossentropy(self.yPred,y)
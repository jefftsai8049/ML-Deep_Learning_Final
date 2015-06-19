__author__ = 'root'

import numpy as np
import theano.tensor as T
import theano

class mlp:
    def __init__(self,x,y,layerNumber,parameters = None):
        randomSeed = np.random.RandomState(8887)
        if not parameters == None:
            self.inputLayer = hiddenLayer(randomSeed,x,layerNumber[0],layerNumber[1],parameters[0],parameters[1])
            self.hiddenLayer = hiddenLayer(randomSeed,self.inputLayer.z,layerNumber[1],layerNumber[2],parameters[2],parameters[3])
            # self.hiddenLayer2 = hiddenLayer(randomSeed,self.hiddenLayer.z,layerNumber[2],layerNumber[3],parameters[4],parameters[5])
            self.outputLayer = outputLayer(randomSeed,self.hiddenLayer.z,layerNumber[2],layerNumber[3],parameters[4],parameters[5])
        else:
            self.inputLayer = hiddenLayer(randomSeed,x,layerNumber[0],layerNumber[1])
            self.hiddenLayer = hiddenLayer(randomSeed,self.inputLayer.z,layerNumber[1],layerNumber[2])
            # self.hiddenLayer2 = hiddenLayer(randomSeed,self.hiddenLayer.z,layerNumber[2],layerNumber[3])
            self.outputLayer = outputLayer(randomSeed,self.hiddenLayer.z,layerNumber[2],layerNumber[3])

        self.parameters = self.inputLayer.parameters+self.hiddenLayer.parameters+self.outputLayer.parameters
        # self.parameters = self.inputLayer.parameters+self.hiddenLayer.parameters+self.outputLayer.parameters

    def predict(self,x):
        self.inputLayer.predict(x)
        self.hiddenLayer.predict(self.inputLayer.z)
        # self.hiddenLayer2.predict(self.hiddenLayer.z)
        self.outputLayer.predict(self.hiddenLayer.z)

        return self.outputLayer.yPred


class hiddenLayer:
    def __init__(self,randomSeed,x,inLayer,outLayer,W=None,b=None,activationFunction = T.nnet.sigmoid):
        lower = -np.sqrt(6. / (inLayer + outLayer))
        upper = np.sqrt(6. / (inLayer + outLayer))
        if W is None:
            WValue = np.array(randomSeed.uniform(low=lower,high=upper,size=(inLayer,outLayer)),dtype=theano.config.floatX)
            W = theano.shared(value=WValue,name="W",borrow=True)
            # W = W.get_value()

        if b is None:
            bValue = np.zeros((outLayer,),dtype=theano.config.floatX)
            b= theano.shared(value=bValue,name="b",borrow=True)
        self.W = W
        self.b = b
        self.activationFunction = activationFunction
        a = T.dot(x,self.W)+self.b
        z = self.activationFunction(a)
        self.z = z
        self.parameters = [self.W, self.b]

    def predict(self,x):
        a = T.dot(x,self.W)+self.b
        z = self.activationFunction(a)
        self.z = z

class outputLayer:
    def __init__(self,randomSeed,x,inLayer,outLayer,W=None,b=None,activationFunction = T.nnet.softmax):
        lower = 0
        upper = 0
        if W is None:
            WValue = np.zeros((inLayer,outLayer),dtype=theano.config.floatX)
            W = theano.shared(value=WValue,name="W",borrow=True)
            # W = W.get_value()

        if b is None:
            bValue = np.zeros((outLayer,),dtype=theano.config.floatX)
            b= theano.shared(value=bValue,name="b",borrow=True)

        self.W = W
        self.b = b
        self.activationFunction = activationFunction
        a = T.dot(x,self.W)+self.b
        y = self.activationFunction(a)
        self.parameters = [self.W, self.b]
        self.yPred = y

    def predict(self,x):
        a = T.dot(x,self.W)+self.b
        y = self.activationFunction(a)
        self.yPred = y

    def costFunction(self,y):
        # return T.mean(T.ceil((self.yPred-y)/48.))
        return T.mean(T.nnet.categorical_crossentropy(self.yPred,y))

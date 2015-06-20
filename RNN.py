__author__ = 'root'

import numpy as np
import theano
import theano.tensor as T


class RNN:
    def __init__(self,x,layerNumbers,activationFunction):
        self.x = x
        self.activationFunction = activationFunction

        # layerNumbers = [nIn,nHidden,nOut]

        # for recurrent weight
        WHInitial = np.asarray(np.random.normal(size = (layerNumbers[1],layerNumbers[1]),low = -0.01,high = 0.01),dtype = theano.config.floatX)
        self.WH = theano.shared(value = WHInitial,name = "WH")
        # for input layer
        WInInitial = np.asarray(np.random.normal(size = (layerNumbers[0],layerNumbers[1]),low = -0.01,high = 0.01),dtype = theano.config.floatX)
        self.WIn = theano.shared(value = WInInitial,name = "WIn")
        # for output layer
        WOutInitail = np.asarray(np.random.normal(size = (layerNumbers[1],layerNumbers[2]),low = -0.01,high = 0.01),dtype = theano.config.floatX)
        self.WOut= theano.shared(value = WOutInitail,name = "WOut")

        bHInitial = np.zeros((layerNumbers[1],),dtype = theano.config.floatX)
        self.bH = theano.shared(value = bHInitial,name = "bH")

        bInInitial = np.zeros((layerNumbers[0],),dtype = theano.config.floatX)
        self.bIn = theano.shared(value = bInInitial,name = "bIn")

        bOutInitial = np.zeros((layerNumbers[2],),dtype = theano.config.floatX)
        self.bOut = theano.shared(value = bOutInitial,name = "bOut")

        self.parameters = [self.WH, self.WIn, self.WOut, self.bH, self.bIn, self.bOut]

__author__ = 'root'

import numpy as np
import theano
import theano.tensor as T
import os.path
import matplotlib.pyplot as plot
import time

import RNN
import preprocessing

preproc = preprocessing.preprocessing()

sequenceLengthFileName = "data/out/sequence.csv"
sortNoShuffleTrainFileName = "data/out/trainNoShuffle.csv"
map48FileName = "data/out/map.csv"

if not os.path.isfile(sequenceLengthFileName):
    print("Generating sequence file...")
    preproc.generatingSequence(sortNoShuffleTrainFileName,sequenceLengthFileName)

preproc.loadTrainFile(sortNoShuffleTrainFileName)
preproc.load48Map(map48FileName)

layers = [69,128,48]
learningRate = 0.0001

x = T.matrix("x")
y = T.matrix("y")
memoryInitail = T.vector("memoryInitail")

WValue = np.array(np.random.uniform(size=(layers[1],layers[1]),low = -0.01,high = 0.01),dtype = theano.config.floatX)
W = theano.shared(value=WValue,name="W",borrow=True)

WInValue = np.array(np.random.uniform(size=(layers[0],layers[1]),low = -0.01,high = 0.01),dtype = theano.config.floatX)
WIn = theano.shared(value=WInValue,name="WIn",borrow=True)

WOutValue = np.array(np.random.uniform(size=(layers[1],layers[2]),low = -0.01,high = 0.01),dtype = theano.config.floatX)
WOut = theano.shared(value=WOutValue,name="WOut",borrow=True)

def step(xIn,memory,W,WIn,WOut):
    a = T.dot(xIn,WIn)+T.dot(memory,W)
    z = T.nnet.sigmoid(T.dot(a,WOut))
    yGivenX = T.nnet.softmax(z)
    return a,yGivenX

parameters = [W,WIn,WOut]
[a,yPred],_ = theano.scan(step,sequences=x,outputs_info=[memoryInitail,None],non_sequences=parameters)
cost = ((y - yPred) ** 2).sum()


gradient = T.grad(cost,parameters)
updates = [(p, p - learningRate * g) for p, g in zip(parameters, gradient)]
trainModel = theano.function(inputs = [memoryInitail, x, y],outputs=cost,updates=updates,allow_input_downcast=True)

sequence = preproc.loadSequenceData(sequenceLengthFileName)


error = []
iter = []
startTime = time.clock()
for i in range(len(sequence)):
    trX,trY = preproc.loadTrainData(sequence[i])
    error.append(trainModel(np.zeros(shape=(layers[1],)),trX,trY))
    if i%50 == 0:
        print("Iteration {} Cost {}".format(i,error[i]))

    iter.append(i)
endTime = time.clock()
print("Time {}".format((endTime-startTime)/60.))

plot.plot(zip(iter,error))
plot.show()


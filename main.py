__author__ = 'jefftsai'
from preprocessing import preprocessing
import os.path
import numpy as np
import theano.tensor as T
import theano
import mlp
import pickle
import time


# File Name Setting
rawTrainDataFileName = "data/mfcc/train.ark"
rawTrainLabelFileName = "data/label/train.lab"
sortTrainFileName = "data/out/train.csv"
mapRawFileName = "data/conf/phones.60-48-39.map"
map48FileName = "data/out/map.csv"

# Data Preprocessing
preproc = preprocessing()
if not os.path.isfile(sortTrainFileName):
    print("Train file does not exist!\n")
    print("Sorting train file...\n")
    preproc.trainSort(rawTrainDataFileName,rawTrainLabelFileName,sortTrainFileName)
if not os.path.isfile(map48FileName):
    print("Map file does not exist!\n")
    print("Generating map file...\n")
    preproc.generate48toNumberMap(mapRawFileName,map48FileName)

# Load Train Data and Map
preproc.loadTrainFile(sortTrainFileName)
preproc.load48Map(map48FileName)

# Parameters Setting
learningRate = 0.001
L1Reg = 0.00
L2Reg = 0.0001
epochs = 1
batchSize = 1

hiddenLayerSize = 128
inputLayerSize = 39
outputLayerSize = 48

# definition of training model
x = T.dmatrix("x")
y = T.imatrix("y")

rng = np.random.RandomState(8888)

classifier = mlp.mlp(rng = rng,input = x,nIn = inputLayerSize,nHidden = hiddenLayerSize,nOut = outputLayerSize)
cost = classifier.cost(y)+L1Reg*classifier.L1+L2Reg*classifier.L2

params = classifier.parameters
gradient = [T.grad(cost,params) for params in classifier.parameters]

update = [(params,params-learningRate*gradient) for params,gradient in zip(classifier.parameters,gradient)]

trainModel = theano.function(inputs=[x,y],outputs=cost,updates=update)
testModel = theano.function(inputs=[x,y],outputs=classifier.logisticRegressionLayer.p_y_given_x,on_unused_input="ignore")

# Read Training Data

start = time.clock()
for epoch in range(epochs):
    i = 0
    print("Epoch {}".format(epoch+1))
    while 1:
        trX,trY = preproc.loadTrainData(batchSize)
        if i > 112000:
            break
        modelCost = trainModel(trX,trY)
        if i%10000 == 0:
            print("Iteration {}  Cost {}".format(i,modelCost))
        i += 1
    preproc.loadTrainFile(sortTrainFileName)


f = open("mlp.model","wb")
pickle.dump(classifier,f,protocol=pickle.HIGHEST_PROTOCOL)
f.close()
end = time.clock()

print("Time {}".format((end-start)/60.))

# Load Model
# f = open("test.model","rb")
# classifier = pickle.load(f)
# f.close()


# Predict
preproc.loadTrainFile(sortTrainFileName)
error = 0
for i in range(10):
    trX,trY = preproc.loadTrainData(batchSize)
    answer = testModel(trX,trY)
    print(answer)
#     if not (np.argmax(answer,1)[0] == np.argmax(trY,1)[0]):
#         error += 1
# print(error)
__author__ = 'jefftsai'

from preprocessing import preprocessing
import os.path
import numpy as np
import theano.tensor as T
import theano
import pickle
import time


import DNN

# File Name Setting
rawTrainDataFileName = "data/fbank/train.ark"
rawTrainLabelFileName = "data/label/train.lab"
sortTrainFileName = "data/out/train.csv"
sortShuffleTrainFileName = "data/out/trainShuffle.csv"
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
if not os.path.isfile(sortShuffleTrainFileName):
    print("Train file does not exist!\n")
    print("Shuffling train file...\n")
    preproc.load48Map(map48FileName)
    preproc.shuffleTrain(sortTrainFileName,sortShuffleTrainFileName)


# Load Train Data and Map
preproc.loadTrainFile(sortShuffleTrainFileName)
preproc.load48Map(map48FileName)

# Parameters Setting
learningRate = 0.001
L1Reg = 0.00
L2Reg = 0.0001
epochs = 200
batchSize = 128

layerNumber = [69,128,48]
# hiddenLayerSize = 128
# inputLayerSize = 39
# outputLayerSize = 48

# definition of training model
x = T.dmatrix("x")
y = T.dmatrix("y")

model = DNN.mlp(x,y,layerNumber)
# cost = T.mean(T.nnet.categorical_crossentropy(model.outputLayer.yPred,y))
cost = model.outputLayer.costFunction(y)

# gradient = [T.grad(cost,parameters) for parameters in model.parameters]

gradient = T.grad(cost,model.parameters)
updates = [(parameters, parameters - learningRate * g) for parameters, g in zip(model.parameters, gradient)]

yGivenX = model.predict(x)

trainModel = theano.function(inputs=[x,y],outputs=cost,updates=updates,allow_input_downcast=True)
testModel = theano.function(inputs=[x],outputs=yGivenX,allow_input_downcast=True)

# classifier = mlp.mlp(rng = randomSeed,input = x,nIn = inputLayerSize,nHidden = hiddenLayerSize,nOut = outputLayerSize)
# cost = classifier.cost(y)+L1Reg*classifier.L1+L2Reg*classifier.L2
#
# params = classifier.parameters
# gradient = [T.grad(cost,params) for params in classifier.parameters]
#
# update = [(params,params-learningRate*gradient) for params,gradient in zip(classifier.parameters,gradient)]
#
# trainModel = theano.function(inputs=[x,y],outputs=cost,updates=update)
# testModel = theano.function(inputs=[x,y],outputs=classifier.logisticRegressionLayer.p_y_given_x,on_unused_input="ignore")

# Read Training Data

start = time.clock()
for epoch in range(epochs):
    i = 0
    print("Epoch {}".format(epoch+1))
    while 1:
        trX,trY = preproc.loadTrainData(batchSize)
        if i > 1120000./batchSize:
            break
        modelCost = trainModel(trX,trY)
        if i%1000 == 0:
            print("Eopch {} Iteration {}  Cost {}".format(epoch,i,modelCost))
            # print(model.outputLayer.W.get_value())
        # if i%1000 == 0:
            accuracy = 0.0
            trX,trY = preproc.loadTrainData(batchSize)
            answer = testModel(trX)
            accuracy += np.sum(np.argmax(answer,1) == np.argmax(trY,1)) 
            print("accuracy {}%".format(accuracy/batchSize*100.0))
            # print(trX[0],answer[0])
            # print(np.argmax(answer,1),np.argmax(trY,1))
        i += 1
    preproc.loadTrainFile(sortShuffleTrainFileName)
    learningRate *= 0.99

f = open("mlp_gpu.model","wb")
pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
f.close()
end = time.clock()

print("Time {}".format((end-start)/60.))

# Load Model
# f = open("mlp.model","rb")
# model = pickle.load(f)
# f.close()


# Predict
preproc.loadTrainFile(sortShuffleTrainFileName)
error = 0
for i in range(1000):
    trX,trY = preproc.loadTrainData(1)
    answer = testModel(trX)
    # print(np.argmax(answer,1)[0],np.argmax(trY,1)[0])
    if i == 0:
        print(answer,trY)
        print(answer[0][30],trY[0][3])
        print(np.argmax(answer,1)[0],np.argmax(trY,1)[0])
    if not (np.argmax(answer,1)[0] == np.argmax(trY,1)[0]):
        error += 1
print(error)

__author__ = 'jefftsai'

from preprocessing import preprocessing
import os.path
import numpy as np
import theano.tensor as T
import theano
import pickle
import time


import DNN
import RNN

# File Name Setting
rawTrainDataFileName = "data/fbank/train.ark"
rawTrainLabelFileName = "data/label/train.lab"
sortTrainFileName = "data/out/train.csv"
sortShuffleTrainFileName = "data/out/trainShuffle.csv"
sortNoShuffleTrainFileName = "data/out/trainNoShuffle.csv"

mapRawFileName = "data/conf/phones.60-48-39.map"
map48FileName = "data/out/map.csv"

rawTestDataFileName = "data/fbank/test.ark"
TestDataFileName = "data/out/test.csv"
outTestResultFileName = "data/out/testPredict.csv"
outTestLabelFileName = "data/out/testPredictLabel.csv"

modelFileName = "model/mlp_2_57.model"


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
if not os.path.isfile(sortNoShuffleTrainFileName):
    print("Train file does not exist!\n")
    print("Non-Shuffling train file...\n")
    preproc.load48Map(map48FileName)
    preproc.noShuffleTrain(sortTrainFileName,sortNoShuffleTrainFileName)
if not os.path.isfile(TestDataFileName):
    print("Test file does not exist!\n")
    print("Format test data file...\n")
    preproc.formatTestData(rawTestDataFileName,TestDataFileName)


# Load Train Data and Map
preproc.loadTrainFile(sortShuffleTrainFileName)
preproc.load48Map(map48FileName)

# Parameters Setting
learningRate = 0.001
epochs = 400
batchSize = 128

layerNumber = [69,128,128,48]
# hiddenLayerSize = 128
# inputLayerSize = 39
# outputLayerSize = 48

# definition of training model
x = T.matrix("x")
y = T.matrix("y")

# Load Model

if os.path.isfile(modelFileName):
    f = open(modelFileName,"rb")
    oldModel = pickle.load(f)
    f.close()
    model = DNN.mlp(x,y,layerNumber,oldModel.parameters)
else:
    model = DNN.mlp(x,y,layerNumber)

cost = model.outputLayer.costFunction(y)
gradient = T.grad(cost,model.parameters)
updates = [(parameters, parameters - learningRate * g) for parameters, g in zip(model.parameters, gradient)]
yGivenX = model.predict(x)

trainModel = theano.function(inputs=[x,y],outputs=cost,updates=updates,allow_input_downcast=True,on_unused_input='warn')
testModel = theano.function(inputs=[x],outputs=yGivenX,allow_input_downcast=True,on_unused_input='warn')

if not os.path.isfile(modelFileName):
    # Read Training Data
    for epoch in range(epochs):
        start = time.clock()
        i = 0
        print("Epoch {}".format(epoch+1))
        while 1:
            trX,trY = preproc.loadTrainData(batchSize)
            if i > 1120000./batchSize:
                break
            modelCost = trainModel(trX,trY)
            if i%1000 == 0:
                print("Eopch {} Iteration {}  Cost {}".format(epoch,i,modelCost))
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
        end = time.clock()
        print("Time {}".format((end-start)/60.))

    f = open("mlp.model","wb")
    pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


# Load Model
# f = open("mlp.model","rb")
# model = pickle.load(f)
# f.close()


# Validation
preproc.loadTrainFile(sortShuffleTrainFileName)
error = 0
testSamples = 100
for i in range(testSamples):
    trX,trY = preproc.loadTrainData(1)
    answer = testModel(trX)
    if not (np.argmax(answer,1)[0] == np.argmax(trY,1)[0]):
        error += 1
print("Validate Accuracy",format(1-error/float(testSamples)))



# Load test File
if not os.path.isfile(outTestResultFileName):
    print("Predicting test data...")
    preproc.loadTestFile(TestDataFileName)
    # teX = preproc.loadTestData(10)
    # i = 0
    # 166114

    outTestResultFile = open(outTestResultFileName,"w")
    teX = preproc.loadTestData(166114)
    answer = testModel(teX)
    name = []
    preproc.loadTestFile(TestDataFileName)
    for i in range(166114):

        temp = preproc.loadTestSet(1)
        name = temp[0][0]
        outTestResultFile.write(name+",")
        for j in range(len(answer[i])):
            outTestResultFile.write(str(answer[i][j]))
            if not (j == len(answer[i])-1):
                outTestResultFile.write(",")
            else:
                outTestResultFile.write("\n")
    outTestResultFile.close()

if not os.path.isfile(outTestLabelFileName):
    print("Transfer to Label...")
    preproc.testPredictTransfer2Label(outTestResultFileName,outTestLabelFileName)


# model = DNN.mlp(x,y,layerNumber)
# cost = model.outputLayer.costFunction(y)
# gradient = T.grad(cost,model.parameters)
# updates = [(parameters, parameters - learningRate * g) for parameters, g in zip(model.parameters, gradient)]
# yGivenX = model.predict(x)
#
# trainModel = theano.function(inputs=[x,y],outputs=cost,updates=updates,allow_input_downcast=True,on_unused_input='warn')
# testModel = theano.function(inputs=[x],outputs=yGivenX,allow_input_downcast=True,on_unused_input='warn')



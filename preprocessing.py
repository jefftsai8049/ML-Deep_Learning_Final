__author__ = 'jefftsai'
import os.path
import numpy as np
import random

class preprocessing:
    def __init__(self):
        return None

    def trainSort(self, trainDataFileName, trainLabelFileName, outFileName):

        if not(os.path.isfile(trainDataFileName) and os.path.isfile(trainLabelFileName)):
            print("Train data and train label file does not exist!\n")
            return False

        dataFile = open(trainDataFileName,"r")
        labelFile = open(trainLabelFileName,"r")
        outFile = open(outFileName,"w")

        while 1:
            dataStr = dataFile.readline()
            if dataStr == "":
                break
            data = dataStr.split()

            while 1:
                labelStr = labelFile.readline()
                if labelStr == "":
                    # read again
                    labelFile = open(trainLabelFileName,"r")
                else:
                    label = labelStr.split(",")
                    if data[0] == label[0]:
                        label[1] = label[1][:-1]
                        out = label[0:]+data[1:]
                        break
            outFile.write(",".join(out)+"\n")
        dataFile.close()
        labelFile.close()
        outFile.close()

    def shuffleTrain(self, trainDataFileName, outFileName):
        if not(os.path.isfile(trainDataFileName)):
            print("Train data and file does not exist!\n")
            return False

        dataFile = open(trainDataFileName,"r")
        outFile = open(outFileName,"w")

        while 1:
            if dataFile.read(1) == "":
                break
            else:
                dataFile.seek(dataFile.tell()-1)

            dataStr = []
            for i in range(1000):
                text = dataFile.readline()
                if text == "":
                    break
                else:
                    data = text.split(",")
                    data[1] = str(self.map[data[1]])
                    dataStr.append(",".join(data))
            random.shuffle(dataStr)

            for j in range(len(dataStr)):
                outFile.write(dataStr[j])

        dataFile.close()
        outFile.close()

    def noShuffleTrain(self, trainDataFileName, outFileName):
        if not(os.path.isfile(trainDataFileName)):
            print("Train data and file does not exist!\n")
            return False

        dataFile = open(trainDataFileName,"r")
        outFile = open(outFileName,"w")

        while 1:
            if dataFile.read(1) == "":
                break
            else:
                dataFile.seek(dataFile.tell()-1)

            dataStr = []
            for i in range(1000):
                text = dataFile.readline()
                if text == "":
                    break
                else:
                    data = text.split(",")
                    data[1] = str(self.map[data[1]])
                    dataStr.append(",".join(data))


            for j in range(len(dataStr)):
                outFile.write(dataStr[j])

        dataFile.close()
        outFile.close()


    def loadTrainFile(self, trainFileName):
        if not os.path.isfile(trainFileName):
            print("Train set file does not exist!\n")
            return False
        self.trainFile = open(trainFileName,"r")

    def readTrainSet(self, inNumber):
        trainSet = []
        for i in range(inNumber):
            dataStr = self.trainFile.readline()
            if dataStr =="":
                return False
            dataStr = dataStr[:-1]
            data = dataStr.split(",")
            for i in range(len(data)-2):
                data[i+2] = float(data[i+2])
            trainSet.append(data)
        return trainSet

    def splitTrainData(self, data):
        x = []
        y = []
        
        for i in range(len(data)):
            y.append(data[i][1])
            x.append(data[i][2:])
        return np.asarray(x,dtype=np.float64),y

    def mapLabelList(self, labelList, map):

        y = []
        temp = [0]*48
        for i in range(len(labelList)):
            labelList[i] = int(labelList[i])
            temp[labelList[i]] = 1
            y.append(temp)
            temp = [0]*48
        return np.asarray(y,dtype=np.int32)


    def generate48toNumberMap(self, inMapFileName, outMapFileName):
        mapData = self.loadRawMap(inMapFileName)
        outMapFile = open(outMapFileName,"w")

        set = []
        temp = []
        j=0
        for i in range(len(mapData)):
            if mapData[i][1] not in temp:
                temp.append(mapData[i][1])
                set.append(temp)
                outMapFile.write(mapData[i][1]+","+str(j)+"\n")
                j += 1
        print(temp)
        outMapFile.close()

    def load48Map(self, fileName):
        if not os.path.isfile(fileName):
            print("48 map file load failed!\n")        
            return False
        mapFile = open(fileName,"r")
        set = []
        while 1:
            mapStr = mapFile.readline()
            if mapStr == "":
                break
            temp = mapStr.split(",")
            temp[1] = int(temp[1])
            set.append(temp)
        self.map = dict(set)


    def loadTrainData(self,number):
        data = self.readTrainSet(number)
        if data == False:
            return np.empty,np.empty
        trX,trY = self.splitTrainData(data)
        trY = self.mapLabelList(trY,self.map)
        return trX,trY

    def loadRawMap(self, fileName):
        if not os.path.isfile(fileName):
            print("Map file load failed!\n")
            return False
        file = open(fileName,"r")
        set = []
        allSet = []
        while 1:
            mapData = file.readline()
            if mapData == "":
                break
            set = mapData[:-1].split()
            # print(set)
            allSet.append(set)
        file.close()
        return allSet

    def formatTestData(self,inFileName,outFileName):
        if not os.path.isfile(inFileName):
            print("Test raw file does not exist!\n")
            return False
        inFile = open(inFileName,"r")
        outFile = open(outFileName,"w")
        while 1:
            inText = inFile.readline()
            if inText == "":
                break
            inData = inText.split()
            outFile.write(",".join(inData)+"\n")

        inFile.close()
        outFile.close()

    def loadTestFile(self,inFile):
        if not os.path.isfile(inFile):
            print("Test set file does not exist!\n")
            return False
        self.testFile = open(inFile,"r")

    def loadTestSet(self,inNumber):
        testSet = []
        for i in range(inNumber):
            dataStr = self.testFile.readline()
            if dataStr =="":
                return testSet
            dataStr = dataStr[:-1]
            data = dataStr.split(",")
            for i in range(len(data)-2):
                data[i+1] = float(data[i+1])
            testSet.append(data)
        return testSet

    def loadTestData(self,inNumber):
        testSet = self.loadTestSet(inNumber)
        teX = []
        for i in range(len(testSet)):
            if len(testSet[i]) > 0:
                teX.append(testSet[i][1:])
            else:
                break
        return np.asarray(teX,dtype=np.float64)

    def generatingSequence(self, inFileName, outFileName):
        if not os.path.isfile(inFileName):
            print("No train file!")
            return
        else:
            inFile = open(inFileName,"r")
            outFile = open(outFileName,"w")

            sequenceLength = 0
            oldname = "faem0"
            oldVoice = "sil1392"
            while 1:
                inText = inFile.readline()
                if inText is "":
                    outFile.write(str(sequenceLength)+"\n")
                    print(sequenceLength)
                    break
                inData = inText.split(",")
                [name,voice,step] = inData[0].split("_")
                if (name == oldname) and (voice == oldVoice):
                    sequenceLength += 1
                else:
                    if sequenceLength != 0:
                        outFile.write(str(sequenceLength)+"\n")
                        print(sequenceLength)
                    sequenceLength = 1
                oldname = name
                oldVoice = voice

        inFile.close()
        outFile.close()


    def loadSequenceData(self,inFileName):
        if not os.path.isfile(inFileName):
            print("No sequence file!")
            return
        else:
            inFile = open(inFileName,"r")
            sequence = []
            while 1:
                inText = inFile.readline()
                if inText == "":
                    break
                inData = int(inText)
                sequence.append(inData)
            return sequence
        inFile.close

    def testPredictTransfer2Label(self,inFileName,outFileName):
        if not os.path.isfile(inFileName):
            print("No test predict file name!")
            return
        else:
            inFile = open(inFileName,"r")
            outFile = open(outFileName,"w")
            while 1:
                inText = inFile.readline()
                if inText == "":
                    break
                inData = inText.split(",")[1:]
                name = inText.split(",")[0]
                val = []
                for i in range(len(inData)):
                    val.append(float(inData[i]))

                print(val.index(max(val)))
                outFile.write(name+","+str(val.index(max(val)))+"\n")
            inFile.close()
            outFile.close()


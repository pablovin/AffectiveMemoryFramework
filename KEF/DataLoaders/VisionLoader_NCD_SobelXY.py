# -*- coding: utf-8 -*-

import IDataLoader

import os
import cv2
import numpy
import datetime
import random

from keras.utils import np_utils

from KEF.Models import Data



class VisionLoader_NCD(IDataLoader.IDataLoader):
    numberOfAugmentedSamples = 10
    framesPerSecond = 25
    stride = 25

    @property
    def logManager(self):
        return self._logManager

    @property
    def dataTrain(self):
        return self._dataTrain

    @property
    def dataValidation(self):
        return self._dataValidation

    @property
    def dataTest(self):
        return self._dataTest

    @property
    def preProcessingProperties(self):
        return self._preProcessingProperties

    def __init__(self, logManager, preProcessingProperties=None, fps =25, stride=25):

        assert (not logManager == None), "No Log Manager was sent!"

        self._preProcessingProperties = preProcessingProperties

        self._dataTrain = None
        self._dataTest = None
        self._dataValidation = None
        self._logManager = logManager
        self.framesPerSecond = fps
        self.stride=stride


    def dataAugmentation(self, dataPoint):

        samples = []
        samples.append(dataPoint)
        for i in range(self.numberOfAugmentedSamples):
            samples.append(seq.augment_image(dataPoint))

        return numpy.array(samples)

    def preProcess(self, dataLocation, augment=False):

        assert (
        not dataLocation == None or not self.preProcessingProperties == ""), "No data or parameter sent to preprocess!"

        assert (len(self.preProcessingProperties[
                        0]) == 2), "The preprocess have the wrong shape. Right shape: ([imgSize,imgSize])."



        imageSize = self.preProcessingProperties[0]

        grayScale = self.preProcessingProperties[1]


        frame = cv2.imread(dataLocation)


        data = numpy.array(cv2.resize(frame, imageSize))


        if grayScale:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        sobelX = cv2.Sobel(data, cv2.CV_64F, 1, 0, ksize=5)
        sobelY = cv2.Sobel(data, cv2.CV_64F, 0, 1, ksize=5)



        if grayScale:


            data = numpy.expand_dims(data, axis=0)
            sobelX = numpy.expand_dims(sobelX, axis=0)
            sobelY = numpy.expand_dims(sobelY, axis=0)

        else:


            data = numpy.swapaxes(data, 1, 2)
            data = numpy.swapaxes(data, 0, 1)


        data = data.astype('float32')
        sobelX = sobelX.astype('float32')
        sobelY = sobelY.astype('float32')

        data /= 255
        sobelX /= 255
        sobelY /= 255

        data = numpy.array(data)
        sobelX = numpy.array(sobelX)
        sobelY = numpy.array(sobelY)





        return (data, sobelX,sobelY)

    def orderClassesFolder(self, folder):

        classes = os.listdir(folder)

        return classes

    def orderDataFolder(self, folder):

        dataList = os.listdir(folder)




        return dataList


    def loadData(self, dataFolder, augment):

        def shuffle_unison(a, b):
            assert len(a) == len(b)
            p = numpy.random.permutation(len(a))
            return a[p], b[p]

        assert (not dataFolder == None or not dataFolder == ""), "Empty Data Folder!"

        dataX = []
        dataLabels = []
        classesDictionary = []

        gestures = self.orderClassesFolder(dataFolder + "/")
        self.logManager.write("Gestures reading order: " + str(gestures))

        lastImage = None

        numberOfVideos = 0
        gestureNumber = 0
        for e in gestures:

                gestureNumber = gestureNumber+1
                loadedDataPoints = 0
                classesDictionary.append("'" + str(gestureNumber) + "':'" + str(e) + "',")

                numberOfVideos = numberOfVideos + 1

                time = datetime.datetime.now()

                #print dataFolder + "/" + v+"/"+dataPointLocation
                dataFrames = self.orderDataFolder(dataFolder + "/" + e)
                for dataFrame in dataFrames:
                    dataPoint = self.preProcess(dataFolder + "/" + e +"/"+ dataFrame,
                                                augment)


                    dataX.append(dataPoint)
                    dataLabels.append(gestureNumber - 1)
                    loadedDataPoints = loadedDataPoints + 1

                self.logManager.write(
                    "--- Gestures: " + str(e) + "(" + str(loadedDataPoints) + " Data points - " + str(
                        (datetime.datetime.now() - time).total_seconds()) + " seconds" + ")")



        dataLabels = np_utils.to_categorical(dataLabels, gestureNumber)

        dataX = numpy.array(dataX)


        print "Shape Labels:", dataLabels.shape
        print "Shape DataX:", dataX.shape



        dataX, dataLabels = shuffle_unison(dataX,dataLabels)


        dataX = numpy.array(dataX)
        dataLabels = numpy.array(dataLabels)

        return Data.Data(dataX, dataLabels, classesDictionary)

    def loadTrainData(self, dataFolder, augmentData=False):
        self.logManager.newLogSession("Training Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTrain = self.loadData(dataFolder, augmentData)

        self.logManager.write("Total data points: " + str(len(self.dataTrain.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTrain.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTrain.dataY).shape))
        self.logManager.endLogSession()

    def loadTestData(self, dataFolder, augmentData=False):
        self.logManager.newLogSession("Testing Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTest = self.loadData(dataFolder, augmentData)
        self.logManager.write("Total data points: " + str(len(self.dataTest.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTest.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTest.dataY).shape))
        self.logManager.endLogSession()

    def loadValidationData(self, dataFolder, augmentData=False):
        self.logManager.newLogSession("Validation Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataValidation = self.loadData(dataFolder, augmentData)
        self.logManager.write("Total data points: " + str(len(self.dataValidation.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataValidation.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataValidation.dataY).shape))
        self.logManager.endLogSession()

    def saveData(self, folder):
        pass

    def shuffleData(self, dataX, dataY):

        positions = []
        for p in range(len(dataX)):
            positions.append(p)

        random.shuffle(positions)

        newInputs = []
        newOutputs = []
        for p in positions:
            newInputs.append(dataX[p])
            newOutputs.append(dataY[p])

        return (newInputs, newOutputs)

    def loadTrainTestValidationData(self, folder, percentage):
        pass

    def loadNFoldValidationData(self, folder, NFold):
        pass
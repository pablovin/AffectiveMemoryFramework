# -*- coding: utf-8 -*-


"""Specific Loader for frames from the FER+ corpus


 Author: Pablo Barros
 Created on: 02.05.2018
 Last Update: 16.06.2018



"""


import IDataLoader

import os
import cv2
import numpy
import datetime
import random

from KEF.Models import Data

from imgaug import augmenters as iaa

st = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    # st(iaa.Add((-10, 10), per_channel=0.5)),
    # st(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
    # st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
    st(iaa.Affine(
        scale={"x": (0.9, 1.10), "y": (0.9, 1.10)},
        # scale images to 80-120% of their size, individually per axis
        translate_px={"x": (-5, 5), "y": (-5, 5)},  # translate by -16 to +16 pixels (per axis)
        rotate=(-10, 10),  # rotate by -45 to +45 degrees
        shear=(-3, 3),  # shear by -16 to +16 degrees
        order=3,  # use any of scikit-image's interpolation methods
        cval=(0.0, 1.0),  # if mode is constant, use a cval between 0 and 1.0
        mode="constant"
    )),
], random_order=True)



class FER2013PlusLoader(IDataLoader.IDataLoader):

    numberOfAugmentedSamples = 10

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

    def __init__(self, logManager, preProcessingProperties=None):

        assert (not logManager == None), "No Log Manager was sent!"

        self._preProcessingProperties = preProcessingProperties

        self._dataTrain = None
        self._dataTest = None
        self._dataValidation = None
        self._logManager = logManager

    def preProcess(self, dataLocation, augment):

        assert (
        not dataLocation == None or not self.preProcessingProperties == ""), "No data or parameter sent to preprocess!"

        assert (len(self.preProcessingProperties[
                        0]) == 2), "The preprocess have the wrong shape. Right shape: ([imgSize,imgSize])."

        image = cv2.imread(dataLocation)

        imageSize = self.preProcessingProperties[0]
        grayScale = self.preProcessingProperties[1]

        data = numpy.array(cv2.resize(image, imageSize))

        if grayScale:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        if augment:
            data = self.dataAugmentation(data)

        if grayScale:

            if augment:
                data = numpy.expand_dims(data, axis=1)
            else:
                data = numpy.expand_dims(data, axis=0)

        else:

            if augment:
                data = numpy.swapaxes(data, 2, 3)
                data = numpy.swapaxes(data, 1, 2)
            else:
                data = numpy.swapaxes(data, 1, 2)
                data = numpy.swapaxes(data, 0, 1)

        data = data.astype('float32')

        data /= 255

        return data

    def orderClassesFolder(self, folder):

        classes = os.listdir(folder)

        return classes

    def orderDataFolder(self, folder):

        dataList = os.listdir(folder)

        return dataList

    def dataAugmentation(self, dataPoint):


        samples = []
        samples.append(dataPoint)
        for i in range(self.numberOfAugmentedSamples):
            samples.append(seq.augment_image(dataPoint))

        return numpy.array(samples)


    def loadData(self, dataFolder, labelDirectory, augmentData):

        assert (not dataFolder == None or not dataFolder == ""), "Empty Data Folder!"


        dataX = []
        dataLabels = []
        classesDictionary = []

        labelsFile = open(labelDirectory, "r")

        classesDictionary.append("Neutral:0")
        classesDictionary.append("Happiness:1")
        classesDictionary.append("Surprise:2")
        classesDictionary.append("Sadness:3")
        classesDictionary.append("Anger:4")
        classesDictionary.append("Disgust:5")
        classesDictionary.append("Fear:6")
        classesDictionary.append("Contempt:7")

        time = datetime.datetime.now()
        for line in labelsFile:
            lineSplitted = line.split(",")

            fileName = lineSplitted[0]
            labelDistribution = [float(i) for i in lineSplitted[5:13]]
            dataPoint = self.preProcess(dataFolder + "/"+fileName, augmentData)

            if augmentData:
                for dp in dataPoint:
                    dataX.append(dp)
                    dataLabels.append(labelDistribution)
            else:
                dataX.append(dataPoint)
                dataLabels.append(labelDistribution)


        dataLabels = numpy.array(dataLabels)

        dataX = numpy.array(dataX)


        dataX = dataX.astype('float32')


        dataX, dataLabels = self.shuffleData(dataX, dataLabels)

        self.logManager.write(
            "--- Folder: " + dataFolder + "(" + str(
                (datetime.datetime.now() - time).total_seconds()) + " seconds" + ")")
        dataX = numpy.array(dataX)
        dataLabels = numpy.array(dataLabels)

        return Data.Data(dataX, dataLabels, classesDictionary)

    def loadTrainData(self, dataFolder, labelDirectory, augmentData=False):
        self.logManager.newLogSession("Training Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Labels From: " + labelDirectory)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTrain = self.loadData(dataFolder, labelDirectory, augmentData)
        self.logManager.write("Total data points: " + str(len(self.dataTrain.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTrain.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTrain.dataY).shape))
        self.logManager.write("Label dictionary: " + str(self.dataTrain.labelDictionary))
        self.logManager.endLogSession()

    def loadTestData(self, dataFolder, labelDirectory, augmentData=False):
        self.logManager.newLogSession("Testing Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTest = self.loadData(dataFolder, labelDirectory, augmentData)
        self.logManager.write("Total data points: " + str(len(self.dataTest.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTest.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTest.dataY).shape))
        self.logManager.write("Label dictionary: " + str(self.dataTest.labelDictionary))
        self.logManager.endLogSession()

    def loadValidationData(self, dataFolder, labelDirectory, augmentData=False):
        self.logManager.newLogSession("Validation Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataValidation = self.loadData(dataFolder, labelDirectory, augmentData)
        self.logManager.write("Total data points: " + str(len(self.dataValidation.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataValidation.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataValidation.dataY).shape))
        self.logManager.write("Label dictionary: " + str(self.dataValidation.labelDictionary))
        self.logManager.write("Label dictionary: " + str(self.dataValidation.labelDictionary))
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
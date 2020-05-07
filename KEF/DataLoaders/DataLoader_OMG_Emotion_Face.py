# -*- coding: utf-8 -*-

"""Specific Loader for faces from the OMG-Emotion corpus


 Author: Pablo Barros
 Created on: 20.05.2018
 Last Update: 10.08.2018



"""

import datetime
import os
import cv2
import numpy
from KEF.Models import Data_OMG
from keras.utils import np_utils
import IDataLoader

from random import shuffle

from KEF.Models import Data

import csv

class DataLoader_OMG_Face(IDataLoader.IDataLoader):
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

        classes = sorted(os.listdir(folder))
        for c in classes:
            if c.startswith('.'):
                classes.remove(c)
        return classes

    def orderDataFolder(self, folder):
        import re
        def sort_nicely(l):
            """ Sort the given list in the way that humans expect.
            """

            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            l.sort(key=alphanum_key)
            return l

        dataList = sort_nicely(os.listdir(folder))
        for d in dataList:
            if d.startswith('.'):
                dataList.remove(d)

        return dataList


    def loadData(self, dataFolder, dataFile):

        def shuffle_unison(a, b):
            assert len(a) == len(b)
            p = numpy.random.permutation(len(a))
            return a[p], b[p]

        dataX = []
        dataLabels = []
        classesDictionary = []



        time = datetime.datetime.now()
        with open(dataFile, 'rb') as csvfile:
            #print "Reading From:", dataFile
            # raw_input(("here"))
            reader = csv.reader(csvfile)
            rownum = 0
            for row in reader:
                # print "Row:", row
                if not len(row)==0:
                    if rownum >= 1:

                        video = row[3]
                        utterance = row[4]
                        arousal = row[5]
                        valence = row[6]
                        readFrom = dataFolder+"/"+video+"/video/"+utterance
                        imageFaces = os.listdir(readFrom)
                        shuffle(imageFaces)
                        imageFaces = imageFaces[0:10]

                        for imageName in imageFaces:
                            try:
                                dataPoint = self.preProcess(readFrom+"/"+imageName, False)
                                dataX.append(dataPoint)
                                dataLabels.append([float(arousal), float(valence)])
                            except:
                                print ("problem:",readFrom+"/"+imageName)

                rownum = rownum+1
        dataX = numpy.array(dataX)
        dataLabels = numpy.array(dataLabels)
        self.logManager.write(
            "(" + str(len(dataX)) + " Data points - " + str(
                (datetime.datetime.now() - time).total_seconds()) + " seconds" + ")")

        dataX, dataLabels = shuffle_unison(dataX, dataLabels)

        return Data.Data(dataX, dataLabels, classesDictionary)


    def loadTrainData(self, dataFolder, dataFile):
        self.logManager.newLogSession("Training Data")
        self.logManager.write("Loading From: " + dataFolder)

        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTrain = self.loadData(dataFolder, dataFile)
        self.logManager.write("Total data points: " + str(len(self.dataTrain.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTrain.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTrain.dataY).shape))
        self.logManager.write("Label dictionary: " + str(self.dataTrain.labelDictionary))
        self.logManager.endLogSession()

    def loadTestData(self, dataFolder, dataFile):
        self.logManager.newLogSession("Testing Data")
        self.logManager.write("Loading From: " + dataFolder)

        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTest = self.loadData(dataFolder, dataFile)
        self.logManager.write("Total data points: " + str(len(self.dataTest.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTest.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTest.dataY).shape))
        self.logManager.write("Label dictionary: " + str(self.dataTest.labelDictionary))
        self.logManager.endLogSession()

    def loadValidationData(self, dataFolder, dataFile):
        self.logManager.newLogSession("Validation Data")
        self.logManager.write("Loading From: " + dataFolder)

        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataValidation = self.loadData(dataFolder, dataFile)
        self.logManager.write("Total data points: " + str(len(self.dataValidation.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataValidation.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataValidation.dataY).shape))
        self.logManager.write("Label dictionary: " + str(self.dataValidation.labelDictionary))
        self.logManager.endLogSession()

    def saveData(self, folder):
        pass

    def shuffleData(self, folder):
        pass

    def loadTrainTestValidationData(self, folder, percentage):
        pass

    def loadNFoldValidationData(self, folder, NFold):
        pass

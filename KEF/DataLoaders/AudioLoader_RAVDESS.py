# -*- coding: utf-8 -*-
"""Specific Loader for audio from the RAVDESS corpus

Load audio from the RAVDESS corpus. Each wav file is transformed into Melspectrums. We use 3s of audio.


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

from keras.utils import np_utils

from KEF.Models import Data

import librosa



class AudioLoader_RAVDESS(IDataLoader.IDataLoader):
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


    def slice_signal(self, signal, sliceSize, stride=0.5):
        """ Return windows of the given signal by sweeping in stride fractions
            of window
        """

        sliceSize = 16000 * sliceSize

        slices = []
        currentFrame = 0



        while currentFrame+sliceSize < len(signal):
            currentSlide = signal[currentFrame:int(currentFrame+sliceSize)]
            slices.append(currentSlide)
            currentFrame = int(currentFrame+sliceSize*stride)

        return numpy.array(slices)



    def preProcess(self, dataLocation, augment=False):

        assert (
        not dataLocation == None or not self.preProcessingProperties == ""), "No data or parameter sent to preprocess!"

        assert (len(self.preProcessingProperties[
                        0]) == 2), "The preprocess have the wrong shape. Right shape: ([imgSize,imgSize])."


        wav_data, sr = librosa.load(dataLocation, mono=True, sr=16000)

        signals = self.slice_signal(wav_data, 3, 1)

        signals2 = []
        for wav_data2 in signals:
            mel = librosa.feature.melspectrogram(y=wav_data2, sr=sr, n_mels=96)

            self.originalspectogramSize = (94, 96) # for 3 seconds
            mel = numpy.array(cv2.resize(mel, (96, 96)))

            mel = numpy.expand_dims(mel, axis=0)

            signals2.append(mel)


        return numpy.array(signals2)



    def orderClassesFolder(self, folder):

        classes = os.listdir(folder)

        return classes

    def orderDataFolder(self, folder):

        dataList = os.listdir(folder)


        dataList = sorted(dataList, key=lambda x: int(x.split(".")[0]))

        return dataList


    def loadData(self, dataFolder):

        def shuffle_unison(a, b):
            assert len(a) == len(b)
            p = numpy.random.permutation(len(a))
            return a[p], b[p]

        assert (not dataFolder == None or not dataFolder == ""), "Empty Data Folder!"

        dataX = []
        dataLabels = []
        classesDictionary = []

        emotions = self.orderClassesFolder(dataFolder + "/")
        self.logManager.write("Emotions reading order: " + str(emotions))

        emotionNumber = 0
        for e in emotions:
          if not e=="Surprised":
            emotionNumber = emotionNumber+1
            loadedDataPoints = 0
            classesDictionary.append("'" + str(emotionNumber) + "':'" + str(e) + "',")

            time = datetime.datetime.now()

            audios = os.listdir(dataFolder + "/" + e+"/")

            for audio in audios:

                dataPoint = self.preProcess(dataFolder + "/" + e+"/"+audio)

                for audio in dataPoint:
                    dataX.append(audio)
                    dataLabels.append(emotionNumber - 1)
                    loadedDataPoints = loadedDataPoints + 1


            self.logManager.write(
                "--- Emotion: " + str(e) + "(" + str(loadedDataPoints) + " Data points - " + str(
                    (datetime.datetime.now() - time).total_seconds()) + " seconds" + ")")



        dataLabels = np_utils.to_categorical(dataLabels, emotionNumber)

        dataX = numpy.array(dataX)


        dataX, dataLabels = shuffle_unison(dataX,dataLabels)

        dataX = numpy.array(dataX)
        dataLabels = numpy.array(dataLabels)

        return Data.Data(dataX, dataLabels, classesDictionary)

    def loadTrainData(self, dataFolder):
        self.logManager.newLogSession("Training Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTrain = self.loadData(dataFolder)

        self.logManager.write("Total data points: " + str(len(self.dataTrain.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTrain.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTrain.dataY).shape))
        self.logManager.endLogSession()


    def loadTestData(self, dataFolder):
        self.logManager.newLogSession("Testing Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTest = self.loadData(dataFolder)
        self.logManager.write("Total data points: " + str(len(self.dataTest.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTest.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTest.dataY).shape))
        self.logManager.endLogSession()

    def loadValidationData(self, dataFolder):
        self.logManager.newLogSession("Validation Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataValidation = self.loadData(dataFolder)
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
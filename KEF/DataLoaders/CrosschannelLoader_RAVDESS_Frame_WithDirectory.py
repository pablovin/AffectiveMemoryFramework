# -*- coding: utf-8 -*-

"""Specific Loader for audio/frames from the RAVDESS corpus

Load audio/frames from the RAVDESS corpus. For each video, one frame and one Mel Spectrum is loaded


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

from KEF.Models import DataCrossmodal

import librosa


class CrosschannelLoader_RAVDESS(IDataLoader.IDataLoader):
    numberOfAugmentedSamples = 10
    framesPerSecond = 30
    stride = 25
    timeFrame = 2

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

    @property
    def directoriesTrain(self):
        return self._directoriesTrain

    @property
    def directoriesTest(self):
        return self._directoriesTest

    @property
    def directoriesValidation(self):
        return self._directoriesValidation

    def __init__(self, logManager, preProcessingProperties=None, fps =30, stride=25):

        assert (not logManager == None), "No Log Manager was sent!"

        self._preProcessingProperties = preProcessingProperties

        self._dataTrain = None
        self._dataTest = None
        self._dataValidation = None
        self._logManager = logManager
        self.framesPerSecond = fps
        self.stride=stride


    def slice_signal(self, signal, sliceSize, stride=1):
        """ Return windows of the given signal by sweeping in stride fractions
            of window
        """

        sliceSize = (2 ** 14) * sliceSize
        slices = []
        currentFrame = 0




        while currentFrame+sliceSize < len(signal):
            currentSlide = signal[currentFrame:int(currentFrame+sliceSize)]
            slices.append(currentSlide)
            currentFrame = int(currentFrame+sliceSize*stride)
            #print "Shape Current slide:", len(currentSlide)


        return numpy.array(slices, dtype=numpy.int32)




    def preProcess(self, dataLocation, augment=False):
        pass

    def preProcessImage(self, dataLocation, augment=False):

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

        data = numpy.array(data)

        return data


    def preProcessAudio(self, dataLocation, augment=False):

        assert (
        not dataLocation == None or not self.preProcessingProperties == ""), "No data or parameter sent to preprocess!"

        assert (len(self.preProcessingProperties[
                        0]) == 2), "The preprocess have the wrong shape. Right shape: ([imgSize,imgSize])."



        wav_data, sr = librosa.load(dataLocation, mono=True, sr=16000)


        signals = self.slice_signal(wav_data, 3, 1)
        signals = [wav_data[0:49152]]


        signals2 = []
        for wav_data2 in signals:


            mel = librosa.feature.melspectrogram(y=wav_data2, sr=sr, n_mels=80)

            self.originalspectogramSize = (94, 80) # for 3 seconds
            mel = numpy.array(cv2.resize(mel, (80, 96)))

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


        audioDataFolder = dataFolder[0]
        visionDataFolder = dataFolder[1]

        def shuffle_unison(a, b):
            assert len(a) == len(b)
            p = numpy.random.permutation(len(a))
            return a[p], b[p]

        assert (not dataFolder == None or not dataFolder == ""), "Empty Data Folder!"

        dataX = []
        directories = []
        dataXVideo = []
        dataXAudio = []
        dataLabels = []
        classesDictionary = []

        emotions = self.orderClassesFolder(audioDataFolder + "/")
        self.logManager.write("Emotions reading order: " + str(emotions))





        emotionNumber = 0

        for e in emotions:
           if not e == "Surprised":
            emotionNumber = emotionNumber+1
            loadedDataPoints = 0
            classesDictionary.append("'" + str(emotionNumber) + "':'" + str(e) + "',")

            time = datetime.datetime.now()


            numberOfVideos = 0
            audios = os.listdir(audioDataFolder + "/" + e+"/")

            for audio in audios:


                numberOfVideos = numberOfVideos + 1
                subjectName = audio[0:-4]
                videoFolder = visionDataFolder + "/" + e+ "/" +subjectName+".mp4"
                dataFrames = self.orderDataFolder(videoFolder)

                dataPoint = self.preProcessAudio(audioDataFolder + "/" + e+"/"+audio)

                startFrame = 0

                for audioDT in dataPoint:


                    framesDirectory = dataFrames[startFrame:startFrame+self.framesPerSecond]
                    startFrame = startFrame+self.framesPerSecond
                    oneFrame = framesDirectory[15]
                    frame = self.preProcessImage(videoFolder+"/"+oneFrame,False)

                    dataXAudio.append(audioDT)
                    dataXVideo.append(frame)

                    directories.append(audio)

                    dataLabels.append(emotionNumber - 1)
                    loadedDataPoints = loadedDataPoints + 1


            self.logManager.write(
                "--- Emotion: " + str(e) + "(" + str(loadedDataPoints) + " Data points - " + str(
                    (datetime.datetime.now() - time).total_seconds()) + " seconds" + ")")

        dataLabels = np_utils.to_categorical(dataLabels, emotionNumber)

        dataLabels = numpy.array(dataLabels)
        dataXAudio = numpy.array(dataXAudio)
        dataXVideo = numpy.array(dataXVideo)

        print "Shape Labels:", dataLabels.shape
        print "Len:", len(dataX)

        dataX, dataLabels = shuffle_unison(dataX, dataLabels)<


        return DataCrossmodal.DataCrossmodal(dataXAudio, dataXVideo, dataLabels, classesDictionary), directories

    def loadTrainData(self, dataFolder):
        self.logManager.newLogSession("Training Data")
        self.logManager.write("Loading From: " + dataFolder[0]+" and " + dataFolder[1])
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTrain,self._directoriesTrain = self.loadData(dataFolder)


        self.logManager.write("Total data points: " + str(len(self.dataTrain.dataXAudio)))
        self.logManager.write("Data points Audio shape: " + str(numpy.array(self.dataTrain.dataXAudio).shape))
        self.logManager.write("Data points Vision shape: " + str(numpy.array(self.dataTrain.dataXVideo).shape))


        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTrain.dataY).shape))
        self.logManager.endLogSession()




    def loadTestData(self, dataFolder):
        self.logManager.newLogSession("Testing Data")
        self.logManager.write("Loading From: " + dataFolder[0] + " and " + dataFolder[1])
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTest,self._directoriesTest = self.loadData(dataFolder)
        self.logManager.write("Total data points: " + str(len(self.dataTest.dataXAudio)))
        self.logManager.write("Data points Audio shape: " + str(numpy.array(self.dataTest.dataXAudio).shape))
        self.logManager.write("Data points Vision shape: " + str(numpy.array(self.dataTest.dataXVideo).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTest.dataY).shape))
        self.logManager.endLogSession()
        #raw_input("here")

    def loadValidationData(self, dataFolder):
        self.logManager.newLogSession("Validation Data")
        self.logManager.write("Loading From: " + dataFolder[0] + " and " + dataFolder[1])
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataValidation,self._directoriesValidation = self.loadData(dataFolder)
        self.logManager.write("Total data points: " + str(len(self.dataValidation.dataXAudio)))
        self.logManager.write("Data points Audio shape: " + str(numpy.array(self.dataValidation.dataXAudio).shape))
        self.logManager.write("Data points Vision shape: " + str(numpy.array(self.dataValidation.dataXVideo).shape))
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


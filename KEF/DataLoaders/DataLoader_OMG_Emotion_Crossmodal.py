# -*- coding: utf-8 -*-


"""Specific Loader for face and audio from the OMG-Emotion corpus

Load audio from the face and audio OMG-Emotion corpus. Each wav file is transformed into Melspectrums. We use 3s of audio.


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
import librosa

from random import shuffle

from KEF.Models import DataCrossmodal

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


    def slice_signal(self, signal, seconds=1, sr=16000):
        """ Return windows of the given signal by sweeping in stride fractions
            of window
        """

        slices = []
        stepSize = int((seconds * sr))

        while len(signal) < stepSize:
            signal = numpy.append(signal, 0)


        totalSteps = int(len(signal) / stepSize)


        for i in range(totalSteps):

            slice = signal[int(i * stepSize):int(i * stepSize) + stepSize]

            slices.append(slice)


        return numpy.array(slices)

    def preProcessAudio(self, dataLocation, augment=False):



        wav_data, sr = librosa.load(dataLocation, mono=True, sr=16000)

        audioSeconds = self._preProcessingProperties[2]

        signals = self.slice_signal(wav_data, seconds=audioSeconds, sr=sr)


        signals2 = []
        for wav_data2 in signals:
            mel = librosa.feature.melspectrogram(y=wav_data2, sr=sr, n_mels=96)


            if audioSeconds == 3:
                self.originalspectogramSize = (94, 96) # for 3 seconds
                mel = numpy.array(cv2.resize(mel, (96, 96)))

            elif audioSeconds == 1:

             self.originalspectogramSize = (32, 96) # for 1 second
             mel = numpy.array(cv2.resize(mel, (32, 96)))

            elif audioSeconds == 0.3:
                self.originalspectogramSize = (10, 96) # for 0.3 seconds
                mel = numpy.array(cv2.resize(mel, (10, 96)))

            mel = numpy.expand_dims(mel, axis=0)

            signals2.append(mel)

        return numpy.array(signals2)



    def preProcess(self, dataLocation, augment):

        assert (
        not dataLocation == None or not self.preProcessingProperties == ""), "No data or parameter sent to preprocess!"

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

        dataXVideo = []
        dataXAudio = []
        dataLabels = []
        classesDictionary = []

        audioDataFolder = "/data/datasets/OMG-Emotion/audio_extraced_all/"

        time = datetime.datetime.now()
        lastImage = None
        with open(dataFile, 'rb') as csvfile:

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

                        audioDataPoint = self.preProcessAudio(audioDataFolder + "/" + video + "/audio/" + utterance+"/audio.wav")

                        audioSeconds = self._preProcessingProperties[2]
                        videoFPS = self._preProcessingProperties[3]
                        readingFPS = audioSeconds*videoFPS

                        audioNumber = 0
                        for audioRepresentation in audioDataPoint:
                            dataXAudio.append(audioRepresentation)
                            dataLabels.append([float(arousal), float(valence)])
                            imageFaces = imageFaces[int(audioNumber*readingFPS):int(audioNumber*readingFPS+readingFPS)]
                            shuffle(imageFaces)


                            try:
                                imageName = imageFaces[0]
                                dataPoint = self.preProcess(readFrom + "/" + imageName, False)
                                dataXVideo.append(dataPoint)
                                lastImage = dataPoint
                            except:
                                print "Error:", readFrom + "/" + imageName
                                dataXVideo.append(lastImage)



                rownum = rownum+1
        dataLabels = numpy.array(dataLabels)
        dataXAudio = numpy.array(dataXAudio)
        dataXVideo = numpy.array(dataXVideo)

        self.logManager.write(
            "(" + str(len(dataXVideo)) + " Data points - " + str(
                (datetime.datetime.now() - time).total_seconds()) + " seconds" + ")")



        return DataCrossmodal.DataCrossmodal(dataXAudio, dataXVideo, dataLabels, classesDictionary)


    def loadTrainData(self, dataFolder, dataFile):
        self.logManager.newLogSession("Training Data")
        self.logManager.write("Loading From: " + dataFolder)

        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTrain = self.loadData(dataFolder, dataFile)
        self.logManager.write("Total data points: " + str(len(self.dataTrain.dataXAudio)))
        self.logManager.write("Data points Audio shape: " + str(numpy.array(self.dataTrain.dataXAudio).shape))
        self.logManager.write("Data points Vision shape: " + str(numpy.array(self.dataTrain.dataXVideo).shape))
        self.logManager.write("Label dictionary: " + str(self.dataTrain.labelDictionary))
        self.logManager.endLogSession()

    def loadTestData(self, dataFolder, dataFile):
        self.logManager.newLogSession("Testing Data")
        self.logManager.write("Loading From: " + dataFolder)

        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTest = self.loadData(dataFolder, dataFile)
        self.logManager.write("Total data points: " + str(len(self.dataTest.dataXAudio)))
        self.logManager.write("Data points Audio shape: " + str(numpy.array(self.dataTest.dataXAudio).shape))
        self.logManager.write("Data points Vision shape: " + str(numpy.array(self.dataTest.dataXVideo).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTest.dataY).shape))
        self.logManager.endLogSession()

    def loadValidationData(self, dataFolder, dataFile):
        self.logManager.newLogSession("Validation Data")
        self.logManager.write("Loading From: " + dataFolder)

        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataValidation = self.loadData(dataFolder, dataFile)
        self.logManager.write("Total data points: " + str(len(self.dataValidation.dataXAudio)))
        self.logManager.write("Data points Audio shape: " + str(numpy.array(self.dataValidation.dataXAudio).shape))
        self.logManager.write("Data points Vision shape: " + str(numpy.array(self.dataValidation.dataXVideo).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataValidation.dataY).shape))
        self.logManager.endLogSession()

    def saveData(self, folder):
        pass

    def shuffleData(self, folder):
        pass

    def loadTrainTestValidationData(self, folder, percentage):
        pass

    def loadNFoldValidationData(self, folder, NFold):
        pass

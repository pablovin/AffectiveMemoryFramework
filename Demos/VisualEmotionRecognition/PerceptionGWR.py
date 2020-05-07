# -*- coding: utf-8 -*-
import numpy

from keras.models import load_model, Model, Input

from keras.models import load_model


import metrics

import Standard_GWR

import datetime
import cv2



import warnings

import csv
warnings.filterwarnings("ignore", category=DeprecationWarning)

def warn(*args, **kwargs):
    pass

from keras import backend as K
warnings.warn = warn

# from keras.utils.layer_utils import print_layer_shapes




def preProcess(dataLocation, imageSize, grayScale, fromFile=True):

    if fromFile:
        frame = cv2.imread(dataLocation)
    else:
        frame = dataLocation

    data = numpy.array(cv2.resize(frame, imageSize))

    if grayScale:
       data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
       data = numpy.expand_dims(data, axis=0)

    elif K.image_dim_ordering() == "th":
            data = numpy.swapaxes(data, 1, 2)
            data = numpy.swapaxes(data, 0, 1)

    data = data.astype('float32')

    data /= 255

    data = numpy.array(data)

    return data


class Vision_PerceptionGWR_Dimensional_AffectNet():
    batchSize = 80
    numberOfEpochs = 100

    @property
    def modelName(self):
        return self._modelName

    @property
    def model(self):
        return self._model

    @property
    def logManager(self):
        return self._logManager

    @property
    def plotManager(self):
        return self._plotManager

    @property
    def experimentManager(self):
        return self._experimentManager


    def __init__(self, backend="tf"):
        if backend == "th":
            K.set_image_dim_ordering('th')



    def buildModel(self, dataTrain):


        standardGWR = Standard_GWR.AssociativeGWR()
        standardGWR.initNetwork(dataTrain,1)
        self._model = standardGWR


    def train(self, dataPointsTrain,):
        self.logManager.newLogSession("Training GWR")
        numberOfEpochs = 5             # Number of training epochs
        #insertionThreshold = 0.85       # Activation threshold for node insertion
        insertionThreshold = 0.78  # Activation threshold for node insertion
        learningRateBMU = 0.35           # Learning rate of the best-matching unit (BMU)
        learningRateNeighbors = 0.76    # Learning rate of the BMU's topological neighbors

        self._model.trainAGWR(dataPointsTrain, numberOfEpochs,insertionThreshold,learningRateBMU,learningRateNeighbors)

        filepath = self.experimentManager.modelDirectory + "/savedGWR"
        #self.logManager.newLogSession(("Saving GWR in:", filepath))
        self._model.saveWeights(filepath)
        self.logManager.endLogSession()


    def createGwrBMusCsv(self, dataPoints, labels, saveDirectory):

        self.logManager.write("Creating CSV at: " + saveDirectory)
        bmus,weights = self.model.getBMU(dataPoints)

        savingArray = []
        for i in range(len(dataPoints)):
            savingArray.append([weights[i], labels[i]])

        numpy.save(saveDirectory, savingArray)

    def evaluateGWR(self, dataPoints, labels, modelDirectory):

        bmus, weights = self.model.getBMU(dataPoints)
        return self.evaluate(weights, labels, modelDirectory)


    def obtainRepresentation(self, dataX, modelDirectory):

        savingArray = []

        dataXPreprocessed = []
        for im in dataX:
            dataXPreprocessed.append(preProcess(im, (64,64), grayScale=True,fromFile= False))

        dataXPreprocessed = numpy.array(dataXPreprocessed)

        preTrainedCNN = load_model(modelDirectory,
                                   custom_objects={'fbeta_score': metrics.fbeta_score, 'recall': metrics.recall,
                                                   'precision': metrics.precision, "ccc": metrics.ccc,
                                                   "rmse": metrics.rmse})

        denseLayerOutput = preTrainedCNN.get_layer(name="denseLayer").output

        classifier = Model(inputs=preTrainedCNN.inputs, outputs=[denseLayerOutput])

        results = classifier.predict([dataXPreprocessed], batch_size=self.batchSize)

        return results

    def createCSVFile(self, dataPoints, modelDirectory,saveLocation=None):

        time = datetime.datetime.now()

        itemsAdded = 0

        savingArray = []

        dataX, dataValidation = dataPoints.dataX, dataPoints.dataY

        preTrainedCNN = load_model(modelDirectory,
                                   custom_objects={'fbeta_score': metrics.fbeta_score, 'recall': metrics.recall,
                                                   'precision': metrics.precision, "ccc":metrics.ccc, "rmse":metrics.rmse})

        preTrainedCNN.summary()

        denseLayerOutput = preTrainedCNN.get_layer(name="denseLayer").output

        classifier = Model(inputs=preTrainedCNN.inputs, outputs=[denseLayerOutput])

        for i in range(0, len(dataX), 1000):

            # print "i", i
            # print "Interval", i+49
            if i + 1000 > len(dataX):
                break

            labels = dataValidation[i: i + 1000]
            imgs = []
            for a in range(1000):
                imgs.append(preProcess(dataX[i + a], (64, 64), True))

            imgs = numpy.array(imgs)
            results = classifier.predict([imgs],batch_size=self.batchSize)

            for r in range(len(results)):
                savingArray.append([results[r], labels[r]])
                itemsAdded = itemsAdded + 1

        totalMissingImages = 0
        missingImgs = dataX[i - 1:-1]
        missingLabels = dataValidation[i - 1:-1]

        # print "Missing Labels", missingLabels.shape
        # raw_input("here")

        # print "Total Datax:", dataX.shape
        # print "Total missingImgs:", missingImgs.shape
        # print "Last Index:", i

        while len(missingImgs) < 1000:
            # print "Missing imgs:", missingImgs.shape
            missingImgs = numpy.append(missingImgs, missingImgs[-1])
            totalMissingImages = totalMissingImages + 1

        # print "total missing images:", totalMissingImages

        imgs = []
        for a in range(1000):
            imgs.append(preProcess(missingImgs[a], (64, 64), True))

        results = classifier.predict([imgs], batch_size=self.batchSize)

        results = results[0:-totalMissingImages]

        # print "Results:", results.shape

        for r in range(len(results)):
            #     print "Missing labels:", missingLabels[r]
            savingArray.append([results[r], missingLabels[r]])
            itemsAdded = itemsAdded + 1

        if not saveLocation == None:
            numpy.save(saveLocation, savingArray)
        else:
            return savingArray


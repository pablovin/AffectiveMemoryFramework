# -*- coding: utf-8 -*-

"""Specific Model for categorizing audio from the OMG-Emotion corpus


 Author: Pablo Barros
 Created on: 20.05.2018
 Last Update: 10.08.2018



"""


import numpy
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Lambda, Activation
from keras.models import load_model, Model, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D


from keras.models import load_model
from keras.optimizers import Adamax

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import regularizers


import IModelImplementation
from KEF.Metrics import metrics


from keras import backend as K
K.set_image_dim_ordering('th')



class Audio_CNN_OMG(IModelImplementation.IModelImplementation):
    batchSize = 64
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


    def __init__(self, experimentManager=None, modelName=None, plotManager=None):
        self._logManager = experimentManager.logManager
        self._experimentManager = experimentManager
        self._modelName = modelName
        self._plotManager = plotManager





    def buildModel(self, inputShape):

        self.logManager.newLogSession("Implementing Model: " + str(self.modelName))



        modelLocation = "TrainedModels/Audio_RAVDESS_3s/Model/weights.best.hdf5"

        preTrainedCNN = load_model(modelLocation,
                                custom_objects={'fbeta_score': metrics.fbeta_score, 'recall': metrics.recall,
                                                'precision': metrics.precision})

        preTrainedCNN.summary()

        cnnOutput = preTrainedCNN.get_layer(name="flatten_9").output

        dense = Dense(100, activation="relu", name="denseLayer", kernel_regularizer = regularizers.l1(0.001))(cnnOutput)

        arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(dense)
        valence_output = Dense(units=1, activation='tanh', name='valence_output')(dense)

        self._model = Model(inputs=preTrainedCNN.input, outputs=[arousal_output,valence_output])




        for layer in self.model.layers:
            layer.trainable = False


        self.model.get_layer(name="denseLayer").trainable = True
        self.model.get_layer(name="arousal_output").trainable = True
        self.model.get_layer(name="valence_output").trainable = True



        self.model.summary()


        self.logManager.endLogSession()

    def train(self, dataPointsTrain, dataPointsValidation, dataAugmentation):

        self.logManager.newLogSession("Creating Histogram plot: Training data")
        self.plotManager.createDataArousalValenceHistogram(dataPointsTrain,"train")
        self.logManager.newLogSession("Creating Histogram plot: Validation data")
        self.plotManager.createDataArousalValenceHistogram(dataPointsValidation, "validation")

        self.logManager.newLogSession("Training Model")


        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.optimizerType = "Adamax"



        self.logManager.write("--- Training Optimizer: " + str(self.optimizerType))

        self.logManager.write("--- Training Strategy: " + str(optimizer.get_config()))

        self.logManager.write("--- Training Batchsize: " + str(self.batchSize))

        self.logManager.write("--- Training Number of Epochs: " + str(self.numberOfEpochs))

        self.model.compile(loss="mean_absolute_error",
                           optimizer=optimizer,
                           metrics=['mse', metrics.ccc])

        filepath = self.experimentManager.modelDirectory + "/weights.best.hdf5"

        checkPoint = ModelCheckpoint(filepath, monitor='val_arousal_output_mean_squared_error',
                                     verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_valence_output_mean_squared_error', factor=0.2, patience=5, min_lr=0.0001,verbose=1)

        history_callback = self.model.fit(dataPointsTrain.dataX, [dataPointsTrain.dataY[:,0],dataPointsTrain.dataY[:,1]],
                                              batch_size=self.batchSize,
                                              epochs=self.numberOfEpochs,
                                          validation_data=(dataPointsValidation.dataX, [dataPointsValidation.dataY[:,0],dataPointsValidation.dataY[:,1]]),
                                              shuffle=True, callbacks=[checkPoint, reduce_lr])

        self.logManager.write(str(history_callback.history))
        self.plotManager.createTrainingPlot(history_callback)
        self.logManager.endLogSession()

    def evaluate(self, dataPoints):
        self.logManager.newLogSession("Model Evaluation")
        evaluation = self.model.evaluate(dataPoints.dataX, dataPoints.dataY, batch_size=self.batchSize)
        self.logManager.write(str(evaluation))
        self.logManager.endLogSession()

    def classify(self, dataPoint):
        # Todo
        return self.model.predict_classes(numpy.array([dataPoint]), batch_size=self.batchSize, verbose=0)

    def save(self, saveFolder):

        print "Save Folder:", saveFolder + "/" + self.modelName + ".h5"
        self.model.save(saveFolder + "/" + self.modelName + ".h5")

    def load(self, loadFolder):
        self._model = load_model(loadFolder, custom_objects={'ccc': metrics.ccc})

        self.logManager.write("--- Loaded Model from: " + str(loadFolder))


# -*- coding: utf-8 -*-

"""Build and train the Audio_Channel for the RAVDESS corpus.


 Author: Pablo Barros
 Created on: 02.05.2018
 Last Update: 16.06.2018



"""


import numpy
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Lambda, Activation
from keras.models import load_model, Model, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.models import load_model
from keras.optimizers import Adagrad

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import IModelImplementation
from KEF.Metrics import metrics


from keras import backend as K
K.set_image_dim_ordering('th')


class Audio_CNN_RAVDESS(IModelImplementation.IModelImplementation):
    batchSize = 32
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


    def buildModel(self, inputShape, numberOfOutputs):

        self.logManager.newLogSession("Implementing Model: " + str(self.modelName))

        def shuntingInhibition(inputs):
            inhibitionDecay = 0.3

            v_c, v_c_inhibit = inputs

            output = (v_c / (inhibitionDecay
                             + v_c_inhibit))

            return output


        numberChannels = 128

        dropout = 0.2

        kernel = 3
        kernelSize = (kernel, kernel)
        hiddenUnits = 50

        auditory_input = Input(shape=inputShape, name="Audio_Network_Input")
        # Conv 1 and 2
        conv1 = Conv2D(filters=numberChannels / 4, kernel_size=kernelSize, padding="same",
                       kernel_initializer="glorot_uniform", name="Audio_conv1")(auditory_input)

        bn1 = BatchNormalization(axis=1)(conv1)
        actv1 = Activation("relu")(bn1)

        mp1 = MaxPooling2D(pool_size=(2, 2), name="Audio_Pool_1")(actv1)
        drop1 = Dropout(dropout)(mp1)

        numberOfConvBlocks = 1

        if numberOfConvBlocks == 2 or numberOfConvBlocks == 3 or numberOfConvBlocks == 4:

            conv5 = Conv2D(numberChannels / 2, kernelSize, padding="same",
                           kernel_initializer="glorot_uniform", name="Audio_conv2")(drop1)
            bn5 = BatchNormalization(axis=1)(conv5)
            actv5 = Activation("relu")(bn5)

            mp3 = MaxPooling2D(pool_size=(2, 2))(actv5)
            drop1 = Dropout(dropout)(mp3)

        elif numberOfConvBlocks == 3 or numberOfConvBlocks == 4:

            conv5 = Conv2D(numberChannels / 2, kernelSize, padding="same",
                           kernel_initializer="glorot_uniform", name="Audio_conv3")(drop1)
            bn5 = BatchNormalization(axis=1)(conv5)
            actv5 = Activation("relu")(bn5)

            mp3 = MaxPooling2D(pool_size=(2, 2))(actv5)
            drop1 = Dropout(dropout)(mp3)

        elif numberOfConvBlocks == 4:
            conv5 = Conv2D(numberChannels, kernelSize, padding="same",
                           kernel_initializer="glorot_uniform", name="Audio_conv5")(drop1)
            bn5 = BatchNormalization(axis=1)(conv5)
            actv5 = Activation("relu")(bn5)

            mp3 = MaxPooling2D(pool_size=(2, 2))(actv5)
            drop1 = Dropout(dropout)(mp3)

        inhibition = True
        if inhibition:
            conv10 = Conv2D(numberChannels / 2, kernelSize, padding="same",
                            kernel_initializer="glorot_uniform", activation="relu",
                            name="Audio_conv10")(drop1)


            conv10_inhibition = Conv2D(numberChannels / 2, kernelSize, padding="same",
                                       kernel_initializer="glorot_uniform", activation="relu",
                                       name="Audio_conv10_inhibition")(drop1)

            v_conv_inhibitted = Lambda(function=shuntingInhibition)([conv10, conv10_inhibition])

            mp4 = MaxPooling2D(pool_size=(2, 2))(v_conv_inhibitted)
            drop1 = Dropout(dropout)(mp4)

        flatten = Flatten()(drop1)

        dense = Dense(hiddenUnits, activation="relu")(flatten)
        drop5 = Dropout(dropout)(dense)

        output = Dense(units=numberOfOutputs, activation='softmax', name='Audio_CategoricalOutput')(drop5)

        self._model = Model(inputs=auditory_input, outputs=output)

        self.model.summary()

        self.logManager.write("--- Plotting and saving the model at: " + str(self.plotManager.plotsDirectory) +
                              "/" + str(self.modelName) + "_plot.png")

        #self.plotManager.creatModelPlot(self.model, str(self.modelName))

        self.logManager.endLogSession()

    def train(self, dataPointsTrain, dataPointsValidation, dataAugmentation):

        self.logManager.newLogSession("Creating Histogram plot: Training data")
        self.plotManager.createCategoricalHistogram(dataPointsTrain,"train")
        self.logManager.newLogSession("Creating Histogram plot: Validation data")
        self.plotManager.createCategoricalHistogram(dataPointsValidation, "validation")

        self.logManager.newLogSession("Training Model")

        optimizer = Adagrad()
        self.optimizerType = "AdaGrad"



        if not self.logManager is None:
            self.logManager.write("Training Strategy: " + str(optimizer.get_config()))

            self.logManager.write("--- Training Optimizer: " + str(self.optimizerType))

            self.logManager.write("--- Training Strategy: " + str(optimizer.get_config()))

            self.logManager.write("--- Training Batchsize: " + str(self.batchSize))

            self.logManager.write("--- Training Number of Epochs: " + str(self.numberOfEpochs))

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=optimizer,
                           metrics=['accuracy', 'categorical_accuracy', metrics.fbeta_score, metrics.recall, metrics.precision])

        filepath = self.experimentManager.modelDirectory + "/weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', mode="min", patience=40)
        reduce_lr = ReduceLROnPlateau(factor=0.5, monitor='val_loss', min_lr=1e-5, patience=2)

        callbacks_list = [checkpoint, early_stopping, reduce_lr]

        history_callback = self.model.fit(dataPointsTrain.dataX, dataPointsTrain.dataY,
                                          batch_size=self.batchSize,
                                          epochs=self.numberOfEpochs,
                                          validation_data=(dataPointsValidation.dataX, dataPointsValidation.dataY),
                                          shuffle=True,
                                          callbacks=callbacks_list)

        if not self.logManager is None:
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
        self._model = load_model(loadFolder, custom_objects={'fbeta_score':metrics.fbeta_score, 'recall':metrics.recall,'precision':metrics.precision})

        self.logManager.write("--- Loaded Model from: " + str(loadFolder))
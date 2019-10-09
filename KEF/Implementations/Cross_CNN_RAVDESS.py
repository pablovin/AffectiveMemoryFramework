# -*- coding: utf-8 -*-
"""Build and train the Cross-channel CNN for the RAVDESS corpus.


 Author: Pablo Barros
 Created on: 02.05.2018
 Last Update: 16.06.2018



"""



import numpy
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Reshape
from keras.models import load_model, Model, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers import concatenate

from keras.models import load_model
from keras.optimizers import  Adagrad

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import IModelImplementation
from KEF.Metrics import metrics


from keras import backend as K
K.set_image_dim_ordering('th')



class Cross_CNN_RAVDESS(IModelImplementation.IModelImplementation):
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




    def loadVisionNetwork(self, modelLocation):

        visionModel = load_model(modelLocation, custom_objects={'fbeta_score': metrics.fbeta_score, 'recall': metrics.recall, 'precision': metrics.precision})


        for layer in visionModel.layers:
            layer.name = "Vision_" + layer.name


        return visionModel.input, visionModel.get_layer(name="Vision_flatten_1").output


    def loadAudioNetwork(self, modelLocation):

        audioModel = load_model(modelLocation, custom_objects={'fbeta_score': metrics.fbeta_score, 'recall': metrics.recall, 'precision': metrics.precision})


        for layer in audioModel.layers:
            layer.name = "Audio_" + layer.name

        return audioModel.input, audioModel.get_layer(name="Audio_flatten_1").output

    def buildModel(self, inputShape, numberOfOutputs):

        self.logManager.newLogSession("Implementing Model: " + str(self.modelName))


        #PretrainedModels

        modelLocationAudio = "/data/trainedNetworks/Audio_3Seconds_Ravdess.hdf5"  # 3seconds

        audioInput, audioOutput = self.loadAudioNetwork(modelLocationAudio)



        modelLocationVision = "/data/trainedNetworks/RAVDESS_Vision_Frame.hdf5" # 1 frame

        visionInput, visionOutput = self.loadVisionNetwork(modelLocationVision)



        denseAudio = Dense(100,activation="relu",name="Audio_Dense_Input")(audioOutput)
        denseVision = Dense(100, activation="relu", name="Vision_Dense_Input")(visionOutput)

        cc_reshape_audio = Reshape((1, 10, 10))(denseAudio)
        cc_reshape_vision = Reshape((1, 10, 10))(denseVision)

        cc_concatenate = concatenate([cc_reshape_audio, cc_reshape_vision], axis=1)

        cc_conv_1 = Conv2D(filters=16, kernel_size=(5, 5), padding='same',
                           kernel_initializer='glorot_uniform',
                           name="cc_conv_1")(cc_concatenate)

        a_bn2 = BatchNormalization()(cc_conv_1)

        a_actv1 = Activation('relu')(a_bn2)
        a_mp2 = MaxPooling2D(pool_size=(2,2), name="cc_Pool")(a_actv1)

        flattenCC = Flatten(name="cc_Flatten")(a_mp2)


        dense = Dense(200, activation="relu", name="denseLayer")(flattenCC)

        drop5 = Dropout(0.25, name="dropout_output")(dense)

        output = Dense(units=numberOfOutputs, activation='softmax', name='Output')(drop5)

        self._model = Model(inputs=[visionInput, audioInput], outputs=output)

        # for layer in self.model.layers:
        #     layer.trainable = False
        #
        # self.model.get_layer(name="denseLayer").trainable = True
        # self.model.get_layer(name="Output").trainable = True

        self.model.summary()

        self.logManager.write("--- Plotting and saving the model at: " + str(self.plotManager.plotsDirectory) +
                              "/" + str(self.modelName) + "_plot.png")

        self.logManager.endLogSession()


    def train(self, dataPointsTrain, dataPointsValidation, dataAugmentation):

        self.logManager.newLogSession("Creating Histogram plot: Training data")
        self.plotManager.createCategoricalHistogram(dataPointsTrain,"train")
        self.logManager.newLogSession("Creating Histogram plot: Validation data")
        self.plotManager.createCategoricalHistogram(dataPointsValidation, "validation")

        self.logManager.newLogSession("Training Model")

        optimizer = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0001)
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
        early_stopping = EarlyStopping(monitor='val_loss', mode="min", patience=60)


        reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.2, patience=10, min_lr=0.00001,verbose=1)

        callbacks_list = [checkpoint, early_stopping, reduce_lr]

        history_callback = self.model.fit([dataPointsTrain.dataXVideo, dataPointsTrain.dataXAudio], dataPointsTrain.dataY,
                                          batch_size=self.batchSize,
                                          epochs=self.numberOfEpochs,
                                          validation_data=([dataPointsValidation.dataXVideo, dataPointsValidation.dataXAudio], dataPointsValidation.dataY),
                                          shuffle=True,
                                          callbacks=callbacks_list)

        if not self.logManager is None:
            self.logManager.write(str(history_callback.history))
            self.plotManager.createTrainingPlot(history_callback)
            self.logManager.endLogSession()

    def evaluate(self, dataPoints):
        self.logManager.newLogSession("Model Evaluation")
        evaluation = self.model.evaluate([dataPoints.dataXVideo, dataPoints.dataXAudio], dataPoints.dataY, batch_size=self.batchSize)
        self.logManager.write(str(evaluation))
        self.logManager.endLogSession()

    def classify(self, dataPoint):
        # Todo
        return self.model.predict(dataPoint, batch_size=self.batchSize, verbose=0)

    def save(self, saveFolder):

        print "Save Folder:", saveFolder + "/" + self.modelName + ".h5"
        self.model.save(saveFolder + "/" + self.modelName + ".h5")

    def load(self, loadFolder):
        self._model = load_model(loadFolder, custom_objects={'fbeta_score':metrics.fbeta_score, 'recall':metrics.recall,'precision':metrics.precision})

        self.logManager.write("--- Loaded Model from: " + str(loadFolder))
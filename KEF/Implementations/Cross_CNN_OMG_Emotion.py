# -*- coding: utf-8 -*-

"""Build and train the Cross Channel for the OMG-Emotion corpus.


 Author: Pablo Barros
 Created on: 02.05.2018
 Last Update: 16.06.2018

"""

import numpy
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Lambda, Activation, Reshape
from keras.models import load_model, Model, Input
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D

from keras.layers import concatenate, add


from keras.models import load_model
from keras.optimizers import  Adagrad

from keras.callbacks import ModelCheckpoint,  ReduceLROnPlateau
from keras import regularizers

import IModelImplementation
from KEF.Metrics import metrics


from keras import backend as K
K.set_image_dim_ordering('th')


# from keras.utils.layer_utils import print_layer_shapes

class Cross_CNN_OMG(IModelImplementation.IModelImplementation):
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

        #armsModel.summary()
        for layer in visionModel.layers:
            layer.name = "Vision_" + layer.name



        return visionModel.input, visionModel.get_layer(name="Vision_flatten_1").output


    def loadAudioNetwork(self, modelLocation):

        audioModel = load_model(modelLocation, custom_objects={'fbeta_score': metrics.fbeta_score, 'recall': metrics.recall, 'precision': metrics.precision})

        #armsModel.summary()
        for layer in audioModel.layers:
            layer.name = "Audio_" + layer.name



        return audioModel.input, audioModel.get_layer(name="Audio_flatten_9").output

    def buildModel(self, inputShape, numberOfOutputs):

        self.logManager.newLogSession("Implementing Model: " + str(self.modelName))

        def shuntingInhibition(inputs):
            inhibitionDecay = 0.5

            v_c, v_c_inhibit = inputs

            output = (v_c / (inhibitionDecay
                             + v_c_inhibit))

            return output

        self.logManager.newLogSession("--- Implementing Model: " + str(self.modelName))

        # # Input Layers
        # auditory_input = Input(shape=[inputShape[0], 1], name="Auditory_Input")
        # visual_input = Input(shape=[inputShape[1], 1], name="Visual_Input")


        #PretrainedModels


        modelLocationAudio = "TrainedModels/Audio_RAVDESS_3s/Model/weights.best.hdf5"


        audioInput, audioOutput = self.loadAudioNetwork(modelLocationAudio)

        modelLocationVision = "TrainedModels/FaceChannel_Vision_FERplus/Model/CNN.h5"

        # preTrainedCNN.summary()
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

        dense = Dense(100, activation="relu", name="denseLayer", kernel_regularizer=regularizers.l1(0.001))(flattenCC)

        arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(dense)
        valence_output = Dense(units=1, activation='tanh', name='valence_output')(dense)

        self._model = Model(inputs=[visionInput, audioInput], outputs=[arousal_output, valence_output])


        self.model.summary()

        self.logManager.write("--- Plotting and saving the model at: " + str(self.plotManager.plotsDirectory) +
                              "/" + str(self.modelName) + "_plot.png")

       # self.plotManager.creatModelPlot(self.model, str(self.modelName))

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

        from keras.losses import logcosh
        self.model.compile(loss="mean_absolute_error",
                           optimizer=optimizer,
                           metrics=['mse', metrics.ccc])

        filepath = self.experimentManager.modelDirectory + "/weights.best.hdf5"

        checkPoint = ModelCheckpoint(filepath, monitor='val_arousal_output_mean_squared_error',
                                     verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_valence_output_mean_squared_error', factor=0.2, patience=5,
                                      min_lr=0.0001, verbose=1)

        callbacks_list = [reduce_lr]

        history_callback = self.model.fit([dataPointsTrain.dataXVideo, dataPointsTrain.dataXAudio], [dataPointsTrain.dataY[:,0],dataPointsTrain.dataY[:,1]],
                                          batch_size=self.batchSize,
                                          epochs=self.numberOfEpochs,
                                          validation_data=([dataPointsValidation.dataXVideo, dataPointsValidation.dataXAudio], [dataPointsValidation.dataY[:,0],dataPointsValidation.dataY[:,1]]),
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
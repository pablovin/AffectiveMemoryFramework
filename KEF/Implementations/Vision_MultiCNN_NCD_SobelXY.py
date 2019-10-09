# -*- coding: utf-8 -*-
import numpy
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Lambda, Activation, Reshape
from keras.models import load_model, Model, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D

from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam, Adamax, Adagrad, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ProgbarLogger, ReduceLROnPlateau, EarlyStopping

from keras.layers import GlobalAveragePooling1D, concatenate, add

from keras import regularizers

#from KEF.CustomObjects import metrics,losses
import IModelImplementation
from KEF.Metrics import metrics


from keras import backend as K
K.set_image_dim_ordering('th')


# from keras.utils.layer_utils import print_layer_shapes

class Vision_MultiCNN_NCD_SobelXY(IModelImplementation.IModelImplementation):
    batchSize = 32
    numberOfEpochs = 50

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



        inputShape = (1,28,28)
        self.logManager.newLogSession("Implementing Model: " + str(self.modelName))

        vision_input_G = Input(shape=inputShape, name="Network_Input_G")
        vision_input_X = Input(shape=inputShape, name="Network_Input_X")
        vision_input_Y = Input(shape=inputShape, name="Network_Input_Y")


        #Channel Sobel X
        conv1X = Conv2D(20, (5, 5), padding="same", kernel_initializer="glorot_uniform", name="X_conv1")(vision_input_X)
        actv1X = Activation("relu")(conv1X)
        mp1X = MaxPooling2D(pool_size=(2, 2))(actv1X)

        conv2X = Conv2D(30, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="X_conv2")(
            mp1X)
        actv2X = Activation("relu")(conv2X)
        mp2X = MaxPooling2D(pool_size=(2, 2))(actv2X)
        flattenX = Flatten(name="X_Flatten")(mp2X)

        # Channel Gray
        conv1G = Conv2D(20, (5,5), padding="same", kernel_initializer="glorot_uniform", name="G_conv1")(
            vision_input_G)
        actv1G = Activation("relu")(conv1G)
        mp1G = MaxPooling2D(pool_size=(2, 2))(actv1G)

        conv2G = Conv2D(30, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="G_conv2")(
            mp1G)
        actv2G = Activation("relu")(conv2G)
        mp2G = MaxPooling2D(pool_size=(2, 2))(actv2G)
        flattenG = Flatten(name="G_Flatten")(mp2G)

        # Channel Sobel Y
        conv1Y = Conv2D(20, (5, 5), padding="same", kernel_initializer="glorot_uniform", name="Y_conv1")(
            vision_input_Y)
        actv1Y = Activation("relu")(conv1Y)
        mp1Y = MaxPooling2D(pool_size=(2, 2))(actv1Y)

        conv2Y = Conv2D(30, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Y_conv2")(
            mp1Y)
        actv2Y = Activation("relu")(conv2Y)
        mp2Y = MaxPooling2D(pool_size=(2, 2))(actv2Y)
        flattenY = Flatten(name="Y_Flatten")(mp2Y)

        concatenated = concatenate([flattenX, flattenG, flattenY], axis=1)

        dense = Dense(100, activation="relu", name="denseLayer")(concatenated)

        output = Dense(units=numberOfOutputs, activation='softmax', name='Output')(dense)

        self._model = Model(inputs=[vision_input_G, vision_input_X, vision_input_Y], outputs=output)


        self.model.summary()



        self.logManager.endLogSession()

    def train(self, dataPointsTrain, dataAugmentation):

        self.logManager.newLogSession("Creating Histogram plot: Training data")
        self.plotManager.createCategoricalHistogram(dataPointsTrain,"train")

        self.logManager.newLogSession("Training Model")


        #optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
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
        early_stopping = EarlyStopping(monitor='val_loss', mode="min", patience=25)
        reduce_lr = ReduceLROnPlateau(factor=0.5, monitor='val_loss', min_lr=1e-5, patience=2)

        callbacks_list = [checkpoint, early_stopping, reduce_lr]

        history_callback = self.model.fit([dataPointsTrain.dataX[:,0,:,:,:], dataPointsTrain.dataX[:,1,:,:,:], dataPointsTrain.dataX[:,2,:,:,:]], dataPointsTrain.dataY,
                                          batch_size=self.batchSize,
                                          epochs=self.numberOfEpochs,
                                          validation_split=0.4,
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
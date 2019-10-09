# -*- coding: utf-8 -*-

"""Build and train the Face Channel for the FER+ corpus.


 Author: Pablo Barros
 Created on: 02.05.2018
 Last Update: 16.06.2018

"""


import matplotlib
matplotlib.use('Agg')

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Input
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import  Adamax
from keras.callbacks import ModelCheckpoint,  EarlyStopping , ReduceLROnPlateau
from sklearn.metrics import r2_score


from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization

from keras.models import load_model

from keras.models import Model


import numpy
import copy

from keras import backend as K
K.set_image_dim_ordering('th')



import IModelImplementation

class CNN_FER2013(IModelImplementation.IModelImplementation):
    
    
    batchSize = 32
    numberOfEpochs = 150
    
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
    def experimentManager(self):        
        return self._experimentManager         
        
        
    @property
    def plotManager(self):        
        return self._plotManager      

    
    def r2_score(self, y_true, y_pred):
        """Implements r2_score metric from sklearn"""
        return r2_score(y_true, y_pred, multioutput="raw_values")
    

    def hinge_onehot(self, y_true, y_pred):
            y_true = y_true * 2 - 1
            y_pred = y_pred * 2 - 1
    
            return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)    
            
            
    def __init__(self, experimentManager=None, modelName=None, plotManager=None):    
        self._logManager = experimentManager.logManager
        self._experimentManager = experimentManager
        self._modelName = modelName
        self._plotManager = plotManager
        

    def buildModel(self, inputShape, numberOfOutputs):
        
        
        def shuntingInhibition(inputs):
            inhibitionDecay = 0.5
    
            v_c, v_c_inhibit = inputs
    
            output = (v_c / (inhibitionDecay
                                    + v_c_inhibit))
    
            return output
        
        print "Input shape:", inputShape
        
        if not self.logManager is None:
            self.logManager.newLogSession("Implementing Model: " + str(self.modelName))
            

        
        nch = 256

        inputLayer = Input(shape=inputShape, name="Vision_Network_Input")

        #Conv1 and 2
        conv1 = Conv2D(nch / 4, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv1")(inputLayer)
        bn1 = BatchNormalization(axis = 1)(conv1)
        actv1 = Activation("relu")(bn1)


        conv2 = Conv2D(nch / 4, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv2")(actv1)
        bn2 = BatchNormalization(axis = 1)(conv2)
        actv2 = Activation("relu")(bn2)

        mp1 = MaxPooling2D(pool_size=(2, 2))(actv2)
        drop1 = Dropout(0.25)(mp1)


        #Conv 3 and 4
        conv3 = Conv2D(nch / 2, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv3")(drop1)
        bn3 = BatchNormalization(axis = 1)(conv3)
        actv3 = Activation("relu")(bn3)


        conv4 = Conv2D(nch / 2, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv4")(actv3)
        bn4 = BatchNormalization(axis = 1)(conv4)
        actv4 = Activation("relu")(bn4)


        mp2 = MaxPooling2D(pool_size=(2, 2))(actv4)
        drop2 = Dropout(0.25)(mp2)

        #Conv 5 and 6 and 7
        conv5 = Conv2D(nch / 2, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv5")(drop2)
        bn5 = BatchNormalization(axis = 1)(conv5)
        actv5 = Activation("relu")(bn5)


        conv6 = Conv2D(nch / 2, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv6")(actv5)
        bn6 = BatchNormalization(axis = 1)(conv6)
        actv6 = Activation("relu")(bn6)

        conv7 = Conv2D(nch / 2, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv7")(actv6)
        bn7 = BatchNormalization(axis = 1)(conv7)
        actv7 = Activation("relu")(bn7)

        mp3 = MaxPooling2D(pool_size=(2, 2))(actv7)
        drop3 = Dropout(0.25)(mp3)

        #Conv 8 and 9 and 10

        conv8 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv8")(drop3)
        bn8 = BatchNormalization(axis = 1)(conv8)
        actv8 = Activation("relu")(bn8)


        conv9 = Conv2D(nch , (3, 3), padding="same", kernel_initializer="glorot_uniform", name="conv9")(actv8)
        bn9 = BatchNormalization(axis = 1)(conv9)
        actv9 = Activation("relu")(bn9)

        conv10 = Conv2D(nch , (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu", name="conv10")(actv9)


        conv10_inhibition = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu", name="conv10_inhibition")(actv9)


        v_conv_inhibitted = Lambda(function=shuntingInhibition)([conv10, conv10_inhibition])

        mp4 = MaxPooling2D(pool_size=(2, 2))(v_conv_inhibitted)
        drop4 = Dropout(0.25)(mp4)


        flatten = Flatten()(drop4)

        dense = Dense(200, activation="relu")(flatten)
        drop5 = Dropout(0.25)(dense)

        output = Dense(numberOfOutputs, activation="softmax")(drop5)

        model = Model(inputs=inputLayer, outputs=output)

        self._model = model
        

    
        self.model.summary()
        
        if not self.logManager is None:
            self.logManager.endLogSession()
                
                
    def train(self, dataPointsTrain, dataPointsValidation, dataAugmentation):
        
        
        def precision( y_true, y_pred):
            from keras import backend as Kend
            """Precision metric.
            Only computes a batch-wise average of precision.
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = Kend.sum(Kend.round(Kend.clip(y_true * y_pred, 0, 1)))
            predicted_positives = Kend.sum(Kend.round(Kend.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + Kend.epsilon())
            return precision
    
        def recall( y_true, y_pred):
            from keras import backend as Kend
            """Recall metric.
            Only computes a batch-wise average of recall.
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = Kend.sum(Kend.round(Kend.clip(y_true * y_pred, 0, 1)))
            possible_positives = Kend.sum(Kend.round(Kend.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + Kend.epsilon())
            return recall
    
        def fbeta_score( y_true, y_pred, beta=0.5):
            from keras import backend as Kend
            """Computes the F score.
            The F score is the weighted harmonic mean of precision and recall.
            Here it is only computed as a batch-wise average, not globally.
            This is useful for multi-label classification, where input samples can be
            classified as sets of labels. By only using accuracy (precision) a model
            would achieve a perfect score by simply assigning every class to every
            input. In order to avoid this, a metric should penalize incorrect class
            assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
            computes this, as a weighted mean of the proportion of correct class
            assignments vs. the proportion of incorrect class assignments.
            With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
            correct classes becomes more important, and with beta > 1 the metric is
            instead weighted towards penalizing incorrect class assignments.
            """
            if beta < 0:
                raise ValueError('The lowest choosable beta is zero (only precision).')
    
            # If there are no true positives, fix the F score at 0 like sklearn.
            if Kend.sum(Kend.round(Kend.clip(y_true, 0, 1))) == 0:
                return 0
    
            p = precision(y_true, y_pred)
            r = recall(y_true, y_pred)
            bb = beta ** 2
            fbeta_score = (1 + bb) * (p * r) / (bb * p + r + Kend.epsilon())
            return fbeta_score
    
        def fmeasure( y_true, y_pred):
            """Computes the f-measure, the harmonic mean of precision and recall.
            Here it is only computed as a batch-wise average, not globally.
            """
            return fbeta_score(y_true, y_pred, beta=1)
            
            
        if not self.logManager is None:
            self.logManager.newLogSession("Training Model")
        
        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        if not self.logManager is None:
            self.logManager.write("Training Strategy: " + str(optimizer.get_config()))
        
        self.model.compile(loss= "categorical_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy','categorical_accuracy', fbeta_score, recall, precision])
                                   
        filepath=self.experimentManager.modelDirectory + "/weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', mode="min", patience=25)
        reduce_lr = ReduceLROnPlateau(factor = 0.5, monitor='val_loss', min_lr = 1e-5, patience = 2)
        
        callbacks_list = [checkpoint, early_stopping, reduce_lr]
        
        if dataAugmentation:
            
                    # this will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=True,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images                
                        # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(dataPointsTrain.dataX)
        
            # fit the model on the batches generated by datagen.flow()
            history_callback = self.model.fit_generator(datagen.flow(dataPointsTrain.dataX, dataPointsTrain.dataY,shuffle=True,
                                batch_size=self.batchSize),
                                steps_per_epoch=dataPointsTrain.dataX.shape[0] / self.batchSize,
                                epochs=self.numberOfEpochs,
                                validation_data=(dataPointsValidation.dataX, dataPointsValidation.dataY),                                
                                callbacks=callbacks_list)
            
                
        else:        
            history_callback = self.model.fit(dataPointsTrain.dataX, dataPointsTrain.dataY,
                  batch_size=self.batchSize,
                  epochs=self.numberOfEpochs,
                  validation_data=(dataPointsValidation.dataX, dataPointsValidation.dataY),
                  shuffle=True,
                  callbacks=callbacks_list)
                  
            if not self.logManager is None:      
                self.logManager.write(str(history_callback.history))   
                self.logManager.endLogSession()
                
            if not self.plotManager is None:      
                self.plotManager.createTrainingPlot(history_callback, self.modelName)
                
            self.model.load_weights(self.experimentManager.modelDirectory + "/weights.best.hdf5")   
            
            self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy','categorical_accuracy', fbeta_score, recall, precision])                                                   
    
    def getOutputFromConvLayer(self, data, layerName):
        
        somData = copy.deepcopy(data)
        
        dataX = []
        
        for i in range(len(data.dataX)):
        
        
            intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layerName).output)
            output = numpy.array(intermediate_layer_model.predict(numpy.array([data.dataX[i]]))).flatten()
        
            dataX.append(output)
            
        
        dataX = numpy.array(dataX)
        somData.dataX = dataX
        return somData
        
    def evaluate(self, dataPoints):
        if not self.logManager is None:      
            self.logManager.newLogSession("Model Evaluation")
            
        evaluation = self.model.evaluate(dataPoints.dataX, dataPoints.dataY, batch_size=self.batchSize)        
        
        if not self.logManager is None:      
            self.logManager.write(str(evaluation))               
            self.logManager.endLogSession()
            
        return evaluation
        
    
    def classify(self, dataPoint):
        #Todo    
        return self.model.predict_classes(numpy.array([dataPoint]),batch_size=self.batchSize, verbose=0)
        
                
    def save(self, saveFolder):
        
        print "Save Folder:", saveFolder+"/"+self.modelName+".h5"
        self.model.save(saveFolder+"/"+self.modelName+".h5")
    
        
    def load(self, loadFolder):
        def precision( y_true, y_pred):
            from keras import backend as Kend
            """Precision metric.
            Only computes a batch-wise average of precision.
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = Kend.sum(Kend.round(Kend.clip(y_true * y_pred, 0, 1)))
            predicted_positives = Kend.sum(Kend.round(Kend.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + Kend.epsilon())
            return precision
    
        def recall( y_true, y_pred):
            from keras import backend as Kend
            """Recall metric.
            Only computes a batch-wise average of recall.
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = Kend.sum(Kend.round(Kend.clip(y_true * y_pred, 0, 1)))
            possible_positives = Kend.sum(Kend.round(Kend.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + Kend.epsilon())
            return recall
    
        def fbeta_score( y_true, y_pred, beta=0.5):
            from keras import backend as Kend
            """Computes the F score.
            The F score is the weighted harmonic mean of precision and recall.
            Here it is only computed as a batch-wise average, not globally.
            This is useful for multi-label classification, where input samples can be
            classified as sets of labels. By only using accuracy (precision) a model
            would achieve a perfect score by simply assigning every class to every
            input. In order to avoid this, a metric should penalize incorrect class
            assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
            computes this, as a weighted mean of the proportion of correct class
            assignments vs. the proportion of incorrect class assignments.
            With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
            correct classes becomes more important, and with beta > 1 the metric is
            instead weighted towards penalizing incorrect class assignments.
            """
            if beta < 0:
                raise ValueError('The lowest choosable beta is zero (only precision).')
    
            # If there are no true positives, fix the F score at 0 like sklearn.
            if Kend.sum(Kend.round(Kend.clip(y_true, 0, 1))) == 0:
                return 0
    
            p = precision(y_true, y_pred)
            r = recall(y_true, y_pred)
            bb = beta ** 2
            fbeta_score = (1 + bb) * (p * r) / (bb * p + r + Kend.epsilon())
            return fbeta_score
    
        def fmeasure( y_true, y_pred):
            """Computes the f-measure, the harmonic mean of precision and recall.
            Here it is only computed as a batch-wise average, not globally.
            """
            return fbeta_score(y_true, y_pred, beta=1)
            
        self._model = load_model(loadFolder, custom_objects={'fbeta_score':fbeta_score, 'recall':recall,'precision':precision})
        self._model.summary()
        
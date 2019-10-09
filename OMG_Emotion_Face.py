# -*- coding: utf-8 -*-
"""Experiments with the OMG-Emotion Dataset using the Face Channel

More information: Barros, P., Jirak, D., Weber, C., & Wermter, S. (2015). Multimodal emotional state recognition using sequence-dependent deep hierarchical features. Neural Networks, 72, 140-151.

 Parameters:
     baseDirectory (String): Base directory where the experiment will be saved.
     datasetFolderTrain (String): Folder where the audios used for training the model are stored
     datasetFolderTest (String): Folder where the audios used for testing the model are stored
     experimentName (String): Name of the experiment.
     logManager (LogManager):


 Author: Pablo Barros
 Created on: 02.05.2018
 Last Update: 16.06.2018

"""


import matplotlib

matplotlib.use('Agg')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras import backend as K


def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend



def runModel():


    from KEF.Controllers import ExperimentManager

    from KEF.DataLoaders import DataLoader_OMG_Emotion_Face

    from KEF.Implementations import Vision_CNN_OMG_Emotion_Face



    numberOfClasses = 8



    dataDirectory = "/data/OMG_Faces_10Faces/"
    videosDirectory = "/data/datasets/OMG-Emotion/faces_extraced_all/"

    dataFileTrain = "/data/datasets/OMG-Emotion/omg_TrainVideos.csv"

    dataFileValidation = "/data/datasets/OMG-Emotion/omg_ValidationVideos.csv"


    """ Initianize all the parameters and modules necessary
     
         image size: 64,64
    """

    experimentManager = ExperimentManager.ExperimentManager(dataDirectory,"FER+PreTrained", verbose=True)

    grayScale = True
    preProcessingProperties = [(64,64), grayScale]


    """ Loading the training and testing data 
          
    """

    dataLoader = DataLoader_OMG_Emotion_Face.DataLoader_OMG_Face(experimentManager.logManager, preProcessingProperties)
    #

    #
    dataLoader.loadValidationData(videosDirectory, dataFileValidation)

    dataLoader.loadTrainData(videosDirectory, dataFileTrain)



    #""" Creating and tuning the CNN
    #"""
    cnnModel = Vision_CNN_OMG_Emotion_Face.Vision_CNN_OMG_Face(experimentManager, "CNN", experimentManager.plotManager)

    #
    cnnModel.buildModel(dataLoader.dataTrain.dataX.shape[1:])
    ##
    cnnModel.train(dataLoader.dataTrain, dataLoader.dataValidation, False)
    ##

    cnnModel.save(experimentManager.modelDirectory)
    ##

    print "Public Test Evaluation"
    cnnModel.evaluate(dataLoader.dataValidation)

    print "Private Test Evaluation"
    cnnModel.evaluate(dataLoader.dataTest)

set_keras_backend("tensorflow")

print K.backend

if K.backend == "tensorflow":
    import tensorflow as tf

    sess = tf.Session()

    # from keras import backend as K
    K.set_session(sess)

    with tf.device('/gpu:1'):
        runModel()
else:

    runModel()
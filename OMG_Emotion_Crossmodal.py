# -*- coding: utf-8 -*-

"""Experiments with the OMG-Emotion Dataset using the Cross Channel

More Information: Barros, P., Barakova, E., & Wermter, S. (2018). A Deep Neural Model Of Emotion Appraisal. arXiv preprint arXiv:1808.00252.

 Parameters:
     baseDirectory (String): Base directory where the experiment will be saved.
     videosDirectory (String): Folder where the .png images from the dataset are saved
     dataFileTrain (String): File with the list of training samples
     dataFileValidation (String): File with the list of validation samples
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

    from KEF.DataLoaders import DataLoader_OMG_Emotion_Crossmodal

    from KEF.Implementations import Cross_CNN_OMG_Emotion



    numberOfClasses = 8



    dataDirectory = "/data/OMG_Cross_0.3seconds/"
    videosDirectory = "/data/datasets/OMG-Emotion/faces_extraced_all/"

    dataFileTrain = "/data/datasets/OMG-Emotion/omg_TrainVideos.csv"

    dataFileValidation = "/data/datasets/OMG-Emotion/omg_ValidationVideos.csv"


    """ Initianize all the parameters and modules necessary
     
         image size: 64,64
    """

    experimentManager = ExperimentManager.ExperimentManager(dataDirectory,"FER+PreTrained", verbose=True)

    imageSize = 64,64
    grayScale = True
    audioSeconds = 0.3
    videoFPS = 20
    preProcessingProperties = [imageSize, grayScale, audioSeconds, videoFPS]


    """ Loading the training and testing data 
          
    """

    dataLoader = DataLoader_OMG_Emotion_Crossmodal.DataLoader_OMG_Face(experimentManager.logManager, preProcessingProperties)
    #

    #
    dataLoader.loadValidationData(videosDirectory, dataFileValidation)

    dataLoader.loadTrainData(videosDirectory, dataFileTrain)



    #""" Creating and tuning the CNN
    #"""
    cnnModel = Cross_CNN_OMG_Emotion.Cross_CNN_OMG(experimentManager, "CNN", experimentManager.plotManager)


    #
    cnnModel.buildModel((dataLoader.dataTrain.dataXAudio[0].shape, dataLoader.dataTrain.dataXVideo[0].shape),
                        len(dataLoader.dataTrain.labelDictionary))
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
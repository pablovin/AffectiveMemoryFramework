# -*- coding: utf-8 -*-
"""Experiments with the OMG-Emotion Dataset using the Audio Channel

More Information: Barros, P., Weber, C., & Wermter, S. (2016, July). Learning auditory neural representations for emotion recognition. In Neural Networks (IJCNN), 2016 International Joint Conference on (pp. 921-928). IEEE.

 Parameters:
     baseDirectory (String): Base directory where the experiment will be saved.
     videosDirectory (String): Folder where the .wav files from the dataset are saved
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

    from KEF.DataLoaders import DataLoader_OMG_Emotion_Audio

    from KEF.Implementations import Audio_CNN_OMG_Emotion


    dataDirectory = "/data/OMG_Audio_PreTrainedRAVDESS/" # Where the experiment will be saved
    videosDirectory = "/data/datasets/OMG-Emotion/audio_extraced_all/"

    dataFileTrain = "/data/datasets/OMG-Emotion/omg_TrainVideos.csv"

    dataFileValidation = "/data/datasets/OMG-Emotion/omg_ValidationVideos.csv"


    """ Initianize all the parameters and modules necessary
     
         image size: 64,64
    """

    experimentManager = ExperimentManager.ExperimentManager(dataDirectory,"RAVDESS_PreTrained", verbose=True)

    audioSeconds = 0.3
    preProcessingProperties = [audioSeconds]


    """ Loading the training and testing data 
          
    """

    dataLoader = DataLoader_OMG_Emotion_Audio.DataLoader_OMG_Audio(experimentManager.logManager, preProcessingProperties)
    #

    #
    dataLoader.loadValidationData(videosDirectory, dataFileValidation)

    dataLoader.loadTrainData(videosDirectory, dataFileTrain)



    cnnModel = Audio_CNN_OMG_Emotion.Audio_CNN_OMG(experimentManager, "CNN", experimentManager.plotManager)


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


    K.set_session(sess)

    with tf.device('/gpu:1'):
        runModel()
else:

    runModel()
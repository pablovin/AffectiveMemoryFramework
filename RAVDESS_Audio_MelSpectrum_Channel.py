# -*- coding: utf-8 -*-

"""Experiments with the RAVDESS-Audio using Mel Spectrograms

More Information: Barros, P., Weber, C., & Wermter, S. (2016, July). Learning auditory neural representations for emotion recognition. In Neural Networks (IJCNN), 2016 International Joint Conference on (pp. 921-928). IEEE.


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

    from KEF.DataLoaders import AudioLoader_RAVDESS

    from KEF.Implementations import Audio_CNN_RAVDESS

    dataDirectory = "/data/Experiments_RAVDESS_Audio_Mel/"

    """
    Both training and testing data must be stored as:
    
    class1/
         file1.wav
         file2.wav
         ...
    class2/
         file1.wav
         file2.wav
         ... 
    
    """
    datasetFolderTrain = "/data/RAVDESS/audio/train/"
    datasetFolderTest = "/data/RAVDESS/audio/test"


    """ Initianize all the necessary experiment modules     
    """

    experimentManager = ExperimentManager.ExperimentManager(dataDirectory,
                                                            "RAVDESS_Audio_MelSpectrum",
                                                            verbose=True)
    preProcessingProperties = None

    """ Loading the training and testing data 
    """
    dataLoader = AudioLoader_RAVDESS.AudioLoader_RAVDESS(experimentManager.logManager, preProcessingProperties)

    dataLoader.loadTrainData(datasetFolderTrain, augmentData=False)

    dataLoader.loadTestData(datasetFolderTest, augmentData=False)


    """ Building the Network 
    """

    cnnModel = Audio_CNN_RAVDESS.Audio_CNN_RAVDESS(experimentManager, "Vision_Deep_CNN", experimentManager.plotManager)

    cnnModel.buildModel(dataLoader.dataTest.dataX.shape[1:], len(dataLoader.dataTest.labelDictionary))

    """ Training the Network 
        """

    cnnModel.train(dataLoader.dataTrain, dataLoader.dataTest, False)

    cnnModel.save(experimentManager.modelDirectory)

set_keras_backend("tensorflow")

print K.backend

if K.backend == "tensorflow":
    import tensorflow as tf



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    K.set_session(sess)

    with tf.device('/gpu:0'):
        runModel()
else:

    runModel()
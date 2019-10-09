# -*- coding: utf-8 -*-
"""Experiments with the RAVDESS-Crossmodal using Mel Spectrograms and 1 video frame

More Information: Barros, P., & Wermter, S. (2016). Developing crossmodal expression recognition based on a deep neural model. Adaptive behavior, 24(5), 373-396.

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

    from KEF.DataLoaders import CrosschannelLoader_RAVDESS_Frame_WithDirectory

    from KEF.Implementations import Cross_CNN_RAVDESS

    dataDirectory = "/data/barros/RAVDESS_CrossChannel/"

    """
        Both audio training and testing data must be stored as:

        class1/
             file1.wav
             file2.wav
             ...
        class2/
             file1.wav
             file2.wav
             ... 

        """

    datasetFolderAudioTrain = "/data/datasets/RAVDESS/Audio/High/train/"  # RAVDESS_Audio_Train
    datasetFolderAudioTest = "/data/datasets/RAVDESS/Audio/High/test/"  # RAVDESS_Audio_test

    """
            Both video training and testing data must be stored as:

            class1/
                 video1/
                        file1.png
                        file2.png
                 video2/
                        file1.png
                        file2.png
                        file3.png
                 ...
            class2/
                 video1/
                        file1.png
                        file2.png
                 video2/
                        file1.png
                        file2.png
                        file3.png
                 ... 

            """


    datasetFolderVisionTrain = "/data/datasets/RAVDESS/Frames/High/train/"  # RAVDESS_Audio_train
    datasetFolderVisionTest = "/data/datasets/RAVDESS/Frames/High/test/"  # RAVDESS_Audio_test


    """ Initianize all the loading parameters and modules necessary

         image size: 64,64
    """

    experimentManager = ExperimentManager.ExperimentManager(dataDirectory,
                                                            "RADVDESS_CCCNN",
                                                            verbose=True)

    grayScale = True

    preProcessingProperties = [(64, 64), grayScale]

    fps = 90
    stride = 30

    """ Loading the training and testing data 

    """

    dataLoader = CrosschannelLoader_RAVDESS_Frame_WithDirectory.CrosschannelLoader_RAVDESS(experimentManager.logManager, preProcessingProperties, fps, stride)

    dataLoader.loadTrainData([datasetFolderAudioTrain,datasetFolderVisionTrain])

    dataLoader.loadTestData([datasetFolderAudioTest,datasetFolderVisionTest])

    """ Building the Network 
    """

    cnnModel = Cross_CNN_RAVDESS.Cross_CNN_RAVDESS(experimentManager, "Vision_Deep_CNN", experimentManager.plotManager)

    cnnModel.buildModel((dataLoader.dataTest.dataXAudio[0].shape,dataLoader.dataTest.dataXVideo[0].shape), len(dataLoader.dataTest.labelDictionary))


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
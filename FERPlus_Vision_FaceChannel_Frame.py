# -*- coding: utf-8 -*-

"""Experiments with the FER+ Dataset using the Face Channel

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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras import backend as K


def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend


def runModel():
    from KEF.Controllers import ExperimentManager

    # from KEF.DataLoaders import FER2013PlusLoader
    from KEF.DataLoaders import FER2013PlusLoader


    from KEF.Implementations import Vision_CNN_FER2013

    dataDirectory = "/data/VisionFER+/"
    #
    datasetFolderTrain = "/data/datasets/fer2013Plus/Images/FER2013Train/" # FER2013
    datasetFolderTest = "/data/datasets/fer2013Plus/Images/FER2013Test/" # FER2013
    datasetFolderValidation = "/data/datasets/fer2013Plus/Images/FER2013Valid/" # FER2013

    labelFolderTrain = "/data/datasets/fer2013Plus/labels/FER2013Train/label.csv"
    labelFolderTest = "/data/datasets/fer2013Plus/labels/FER2013Test/label.csv"
    labelFolderValidation = "/data/datasets/fer2013Plus/labels/FER2013Valid/label.csv"


    """ Initianize all the parameters and modules necessary

         image size: 64,64
    """

    experimentManager = ExperimentManager.ExperimentManager(dataDirectory,
                                                            "EMOTIW_PReTrained_With_FER_Experiment_GPU_CategoricalCrossentropy_2",
                                                            verbose=True)

    grayScale = True

    preProcessingProperties = [(64, 64), grayScale]

    """ Loading the training and testing data 

    """

    dataLoader = FER2013PlusLoader.FER2013PlusLoader(experimentManager.logManager, preProcessingProperties)
    #
    dataLoader.loadTrainData(datasetFolderTrain, labelFolderTrain)
    #
    dataLoader.loadTestData(datasetFolderTest, labelFolderTest)

    dataLoader.loadValidationData(datasetFolderValidation, labelFolderValidation)


    # """ Creating and tuning the CNN
    # """


    cnnModel = Vision_CNN_FER2013.CNN_FER2013(experimentManager, "CNN", experimentManager.plotManager)

    cnnModel.buildModel(dataLoader.dataTest.dataX.shape[1:], len(dataLoader.dataTest.labelDictionary))
    ##
    cnnModel.train(dataLoader.dataTrain, dataLoader.dataValidation, False)
    ##

    cnnModel.save(experimentManager.modelDirectory)




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
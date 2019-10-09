# -*- coding: utf-8 -*-
"""Experiments with the NCD Dataset using the Multichannel CNN

More information: Barros, P., Magg, S., Weber, C., & Wermter, S. (2014, September). A multichannel convolutional neural network for hand posture recognition. In International Conference on Artificial Neural Networks (pp. 403-410). Springer, Cham.

 Parameters:
     baseDirectory (String): Base directory where the experiment will be saved.
     videosDirectory (String): Folder where the .jpg files from the dataset are saved
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

    from KEF.DataLoaders import VisionLoader_NCD_SobelXY

    from KEF.Implementations import Vision_MultiCNN_NCD_SobelXY

    dataDirectory = "/data/experimentsNCD/"



    datasetFolderTrain = "/data/datasets/20141005_NCD_HandPosture_Barros/images_2"





    """ Initianize all the parameters and modules necessary

         image size: 64,64
    """

    experimentManager = ExperimentManager.ExperimentManager(dataDirectory,
                                                            "RADVDESS_Vision_DeepNetwork_Frame_01",
                                                            verbose=True)

    grayScale = True

    preProcessingProperties = [(28, 28), grayScale]

    fps = 30
    stride = 30

    """ Loading the training and testing data 

    """

    dataLoader = VisionLoader_NCD_SobelXY.VisionLoader_NCD(experimentManager.logManager, preProcessingProperties, fps, stride)


    dataLoader.loadTrainData(datasetFolderTrain, augmentData=False)



    # """ Creating and tuning the CNN
    # """


    cnnModel = Vision_MultiCNN_NCD_SobelXY.Vision_MultiCNN_NCD_SobelXY(experimentManager, "Vision_Deep_CNN", experimentManager.plotManager)


    cnnModel.buildModel(dataLoader.dataTrain.dataX.shape[1:], len(dataLoader.dataTrain.labelDictionary))

    cnnModel.train(dataLoader.dataTrain, False)


    cnnModel.save(experimentManager.modelDirectory)



    print "Private Test Evaluation"
    #cnnModel.evaluate(dataLoader.dataTest)


set_keras_backend("tensorflow")

print K.backend

if K.backend == "tensorflow":
    import tensorflow as tf



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    # from keras import backend as K
    K.set_session(sess)

    with tf.device('/gpu:0'):
        runModel()
else:

    runModel()
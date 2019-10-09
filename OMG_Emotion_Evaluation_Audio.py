
"""Evaluate the Audio files from the OMG-Emotion dataset. It generates the result.csv file.


 Parameters:
     audioFolder (String): Folder where the .wav files from the dataset are saved
     validationCSVDirectory (String): File with the list of validation samples
     saveFolder (String): Folder where the results.csv will be saved
     modelDirectory (String): Location of the trained audio model
     audioSeconds (Float): Duration of the input data



 Author: Pablo Barros
 Created on: 02.05.2018
 Last Update: 16.06.2018

"""


import os
from keras.models import load_model
from KEF.Metrics import metrics
import numpy
import cv2
import csv
from random import shuffle

import librosa
from scipy import signal


audioFolder = "/data/datasets/OMG-Emotion/audio_extraced_all/"


validationCSVDirectory = "/data/datasets/OMG-Emotion/omg_TestVideos_WithLabels.csv"



saveFolder = "/data/OMG_Emotion_Faces_Results/results/"


modelLocation = "TrainedModels/OMG_Emotion_Audio/Model/weights.best.hdf5"


audioSeconds = 3



experimentName = "OMG_Audio_RAVDESS_MEL_TEST_0.3s"


def slice_signal(signal, seconds=1, sr=16000):
    """ Return windows of the given signal by sweeping in stride fractions
        of window
    """

    slices = []

    stepSize = int((seconds * sr))

    totalSteps = int(len(signal) / stepSize)


    for i in range(totalSteps):

        slice = signal[int(i * stepSize):int(i * stepSize) + stepSize]

        slices.append(slice)


    return numpy.array(slices)


def preEmphasis(signal, coeff=0.95):
    x = numpy.array(signal)
    x0 = numpy.reshape(x[0], [1, ])
    diff = x[1:] - coeff * x[:-1]
    concat = numpy.concatenate([x0, diff], 0)
    return concat


def preProcess(dataLocation, augment=False):



    fftsize = 1024
    hop_length = 512

    wav_data, sr = librosa.load(dataLocation, mono=True, sr=16000)



    signals = slice_signal(wav_data, seconds=audioSeconds, sr=sr)



    signals2 = []
    for wav_data2 in signals:
        mel = librosa.feature.melspectrogram(y=wav_data2, sr=sr, n_mels=96)


        if audioSeconds == 3:
            mel = numpy.array(cv2.resize(mel, (96, 96)))

        elif audioSeconds == 1:
            mel = numpy.array(cv2.resize(mel, (32, 96)))

        elif audioSeconds == 0.3:
            mel = numpy.array(cv2.resize(mel, (10, 96)))

        mel = numpy.expand_dims(mel, axis=0)

        signals2.append(mel)

    return numpy.array(signals2)




model = load_model(modelDirectory,
                                 custom_objects={'fbeta_score': metrics.fbeta_score, 'recall': metrics.recall, 'precision': metrics.precision, 'ccc': metrics.ccc})




with open(validationCSVDirectory, 'rb') as csvfile:
    print "Reading From:", validationCSVDirectory

    reader = csv.reader(csvfile)
    rownum = 0

    if not os.path.exists(saveFolder + "/" + experimentName + "/"):
        os.makedirs(saveFolder + "/" + experimentName + "/")
    with open(saveFolder + "/" + experimentName + "/" + "/results.csv", 'a') as the_file:
        text = "video,utterance,arousal,valence"
        the_file.write(text + "\n")


    for row in reader:
        # print "Row:", row
        if rownum >= 1:
            if not len(row)==0:

                video = row[3]
                utterance = row[4]
                arousal = row[5]
                valence = row[6]
                readFrom = audioFolder + "/" + video + "/audio/" + utterance + "/audio.wav"

                if  os.path.exists(readFrom):
                    dataPoint = preProcess(readFrom)

                    meanArousal = []
                    meanValence = []


                    for audio in dataPoint:
                        prediction = model.predict(numpy.array([audio]), batch_size=64, verbose=0)
                        print "Prediciton:", prediction[0]
                        print "Prediciton:", prediction[1]
                        meanArousal.append(float(prediction[0][0]))

                        meanValence.append(float(prediction[1][0]))


                    arousal = numpy.median(numpy.array(meanArousal))
                    valence = numpy.median(numpy.array(meanValence))

                    import math
                    if math.isnan(arousal):
                        arousal = 0

                    if math.isnan(valence):
                        valence = 0

                else:
                    arousal = 0
                    valence = 0

                print "Video: ", video, " Utterance: ", utterance, " A/V: ", str(arousal) + "," + str(valence)
                with open(saveFolder + "/" + experimentName + "/" + "/results.csv", 'a') as the_file:
                    text = video + "," + utterance + "," + str(arousal) + "," + str(valence)
                    the_file.write(text + "\n")
        rownum = rownum+1


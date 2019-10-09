"""Evaluate the Crosschannel model with the OMG-Emotion dataset. It generates the result.csv file.


 Parameters:
     facesFolder (String): Folder where the .png files from the dataset are saved
     audioFolder (String): Folder where the .wav files from the dataset are saved
     validationCSVDirectory (String): File with the list of validation samples
     saveFolder (String): Folder where the results.csv will be saved
     modelDirectory (String): Location of the trained cross model
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
import librosa
from random import shuffle


facesFolder = "/data/datasets/OMG-Emotion/faces_extraced_all/"

audioDataFolder = "/data/datasets/OMG-Emotion/audio_extraced_all/"


validationCSVDirectory = "/data/datasets/OMG-Emotion/omg_TestVideos_WithLabels.csv"


saveFolder = "/data/OMG_Emotion_Crossmodal_Results/results/"


modelDirectory = "TrainedModels/OMG_Emotion_Cross/Model/CNN.h5"



experimentName = "OMG_Cross_FER+Test_FINAL"



model = load_model(modelDirectory,
                                 custom_objects={'fbeta_score': metrics.fbeta_score, 'recall': metrics.recall, 'precision': metrics.precision, 'ccc': metrics.ccc})


def preProcessVideo(dataLocation, augment):

    image = cv2.imread(dataLocation)

    imageSize = 64,64
    grayScale = True

    data = numpy.array(cv2.resize(image, imageSize))

    if grayScale:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    if grayScale:

        if augment:
            data = numpy.expand_dims(data, axis=1)
        else:
            data = numpy.expand_dims(data, axis=0)

    else:

        if augment:
            data = numpy.swapaxes(data, 2, 3)
            data = numpy.swapaxes(data, 1, 2)
        else:
            data = numpy.swapaxes(data, 1, 2)
            data = numpy.swapaxes(data, 0, 1)

    data = data.astype('float32')

    data /= 255

    return data

def slice_signal(signal, seconds=1, sr=16000):
    """ Return windows of the given signal by sweeping in stride fractions
        of window
    """

    slices = []
    stepSize = int((seconds * sr))


    while len(signal) < stepSize:
        signal = numpy.append(signal, 0)



    totalSteps = int(len(signal) / stepSize)


    for i in range(totalSteps):

        slice = signal[int(i * stepSize):int(i * stepSize) + stepSize]

        slices.append(slice)


    return numpy.array(slices)


def preProcessAudio(dataLocation, augment=False):

    wav_data, sr = librosa.load(dataLocation, mono=True, sr=16000)

    audioSeconds = 0.3

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
        if rownum >= 1:
            if not len(row)==0:

                video = row[3]
                utterance = row[4]
                arousal = row[5]
                valence = row[6]

                readFrom = facesFolder + "/" + video + "/video/" + utterance
                imageFaces = os.listdir(readFrom)
                try:
                    audioDataPoint = preProcessAudio(
                        audioDataFolder + "/" + video + "/audio/" + utterance + "/audio.wav")

                    audioSeconds = 0.3
                    videoFPS = 20
                    readingFPS = audioSeconds * videoFPS

                    meanArousal = []
                    meanValence = []

                    audioNumber = 0

                    for audioRepresentation in audioDataPoint:
                        imageFaces = imageFaces[int(audioNumber * readingFPS):int(audioNumber * readingFPS + readingFPS)]
                        shuffle(imageFaces)
                        #
                        try:
                            imageName = imageFaces[0]
                            dataPoint = preProcessVideo(readFrom + "/" + imageName, False)

                            prediction = model.predict([numpy.array([dataPoint]),numpy.array([audioRepresentation])], batch_size=32, verbose=0)

                            valence = float(prediction[1][0])
                            arousal = float(prediction[0][0])

                        except:
                            print "Error:", readFrom + "/" + imageName
                            valence = 0
                            arousal = 0

                        meanArousal.append(arousal)
                        meanValence.append(valence)

                    meanArousal = numpy.array(meanArousal).mean()
                    meanValence = numpy.array(meanValence).mean()

                except:

                    meanArousal = 0
                    meanValence = 0

                print "Video: ", video, " Utterance: ", utterance, " A/V: ", str(meanArousal) + "," + str(meanValence)
                if not os.path.exists(saveFolder + "/" + experimentName + "/"):
                    os.makedirs(saveFolder + "/" + experimentName + "/")
                with open(saveFolder + "/" + experimentName + "/" + "/results.csv", 'a') as the_file:
                    text = video + "," + utterance + "," + str(meanArousal) + "," + str(meanValence)
                    the_file.write(text + "\n")
        rownum = rownum+1


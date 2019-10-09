"""Evaluate the Audio files from the OMG-Emotion dataset. It generates the result.csv file.



 Parameters:
     facesFolder (String): Folder where the .png files from the dataset are saved
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


facesFolder = "/data/datasets/OMG-Emotion/faces_extraced_all/"


validationCSVDirectory = "/data/datasets/OMG-Emotion/omg_TestVideos_WithLabels.csv"



saveFolder = "/data/OMG_Emotion_Faces_Results/results/"


modelDirectory = "TrainedModels/OMG_Emotion_Face/Model/weights.best.hdf5"

experimentName = "OMG_FACE_FER+Test_FINAL_Samples_5_samples"



model = load_model(modelDirectory,
                                 custom_objects={'fbeta_score': metrics.fbeta_score, 'recall': metrics.recall, 'precision': metrics.precision, 'ccc': metrics.ccc})

with open(validationCSVDirectory, 'rb') as csvfile:
    print "Reading From:", validationCSVDirectory
    # raw_input(("here"))
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

                readFrom = facesFolder + "/" + video + "/video/" + utterance
                imageFaces = os.listdir(readFrom)
                #shuffle(imageFaces)
                #imageFaces = imageFaces[0::5]
                meanArousal = []
                meanValence = []

                for imageName in imageFaces:
                    try:
                        print "Read: ", facesFolder + "/" + video + "/video/" +utterance+"/"+ imageName
                        face = cv2.imread(facesFolder + "/" + video + "/video/" +utterance+"/"+ imageName)

                        image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                        image = numpy.array(cv2.resize(image, (64, 64)))

                        #image = numpy.expand_dims(image, axis=-1)
                        image = numpy.expand_dims(image, axis=0)

                        image = image.astype('float32')

                        image /= 255

                        prediction = model.predict(numpy.array([image]), batch_size=32, verbose=0)
                        print "Prediciton:", prediction[0]
                        print "Prediciton:", prediction[1]
                        valence = float(prediction[1][0])
                        #arousal = float(float( (prediction[0][0])+1)/2)
                        arousal = float(prediction[0][0])



                    except:
                        print "problem:", readFrom + "/" + imageName
                        arousal = 0
                        valence = 0
                        
                    meanArousal.append(arousal)
                    meanValence.append(valence)
                    
                meanArousal = numpy.array(meanArousal).mean()
                meanValence = numpy.array(meanValence).mean()

                print "Video: ", video, " Utterance: ", utterance, " A/V: ", str(meanArousal) + "," + str(meanValence)
                if not os.path.exists(saveFolder + "/" + experimentName + "/"):
                    os.makedirs(saveFolder + "/" + experimentName + "/")
                with open(saveFolder + "/" + experimentName + "/" + "/results.csv", 'a') as the_file:
                    text = video + "," + utterance + "," + str(meanArousal) + "," + str(meanValence)
                    the_file.write(text + "\n")
        rownum = rownum+1



from keras.models import Input
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D


from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization

from keras.models import load_model

from keras.models import Model
import tensorflow

originalModel = "/home/pablo/Documents/workspace/AffectiveMemoryFramework/Demos/VisualEmotionRecognition/Trained Networks/Vision/FER2013Plus_Augmented_CNN/Model/CNN.h5"
saveModel = "/home/pablo/Documents/workspace/AffectiveMemoryFramework/Demos/VisualEmotionRecognition/Trained Networks/Vision/FER2013Plus_Augmented_CNN/Model/CNN_Python3.h5"

def shuntingInhibition(inputs):
    inhibitionDecay = 0.5

    v_c, v_c_inhibit = inputs

    output = (v_c / (inhibitionDecay
                     + v_c_inhibit))

    return output

import keras
keras.backend.set_image_data_format("channels_first")


nch = 256
inputShape = (1,64, 64)

inputLayer = Input(shape=inputShape, name="Vision_Network_Input")

#
# Conv1 and 2
conv1 = Conv2D(int(nch / 4), (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv1")(
    inputLayer)
bn1 = BatchNormalization(axis=1)(conv1)
actv1 = Activation("relu")(bn1)

conv2 = Conv2D(int(nch / 4), (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv2")(actv1)
bn2 = BatchNormalization(axis=1)(conv2)
actv2 = Activation("relu")(bn2)

mp1 = MaxPooling2D(pool_size=(2, 2))(actv2)
drop1 = Dropout(0.25)(mp1)

# Conv 3 and 4
conv3 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv3")(drop1)
bn3 = BatchNormalization(axis=1)(conv3)
actv3 = Activation("relu")(bn3)

conv4 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv4")(actv3)
bn4 = BatchNormalization(axis=1)(conv4)
actv4 = Activation("relu")(bn4)

mp2 = MaxPooling2D(pool_size=(2, 2))(actv4)
drop2 = Dropout(0.25)(mp2)

# Conv 5 and 6 and 7
conv5 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv5")(drop2)
bn5 = BatchNormalization(axis=1)(conv5)
actv5 = Activation("relu")(bn5)

conv6 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv6")(actv5)
bn6 = BatchNormalization(axis=1)(conv6)
actv6 = Activation("relu")(bn6)

conv7 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv7")(actv6)
bn7 = BatchNormalization(axis=1)(conv7)
actv7 = Activation("relu")(bn7)

mp3 = MaxPooling2D(pool_size=(2, 2))(actv7)
drop3 = Dropout(0.25)(mp3)

# Conv 8 and 9 and 10

conv8 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv8")(drop3)
bn8 = BatchNormalization(axis=1)(conv8)
actv8 = Activation("relu")(bn8)

conv9 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="conv9")(actv8)
bn9 = BatchNormalization(axis=1)(conv9)
actv9 = Activation("relu")(bn9)

conv10 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu", name="conv10")(
    actv9)

conv10_inhibition = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                           name="conv10_inhibition")(actv9)

v_conv_inhibitted = Lambda(function=shuntingInhibition)([conv10, conv10_inhibition])

mp4 = MaxPooling2D(pool_size=(2, 2))(v_conv_inhibitted)
drop4 = Dropout(0.25)(mp4)

flatten = Flatten()(drop4)

dense = Dense(200, activation="relu")(flatten)
drop5 = Dropout(0.25)(dense)

output = Dense(8, activation="softmax")(drop5)

model = Model(inputs=inputLayer, outputs=output)

model.summary()

model.load_weights(originalModel)
model.save(saveModel)
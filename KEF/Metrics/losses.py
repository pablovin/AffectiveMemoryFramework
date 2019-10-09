import numpy
from keras import backend as K

def hinge_onehot(y_true, y_pred):
    y_true = y_true * 2 - 1
    y_pred = y_pred * 2 - 1

    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)

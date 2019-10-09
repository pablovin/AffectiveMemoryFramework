import numpy


class Data:
    dataX = None
    dataY = None
    labelDictionary = None

    def __init__(self, dataX=None, dataY=None, labelDictionary=None, preprocessingProperties=None):

        self.dataX = dataX
        self.dataY = dataY
        self.labelDictionary = labelDictionary

        self.multi_channel_data(dataX, preprocessingProperties)

    def multi_channel_data(self, dataX, preprocessingProperties):


        if preprocessingProperties[1]:
            channels = 1
        else:
            channels = 3

        self.images = dataX[:, 1]

        self.images = numpy.array([im for im in self.images]).reshape((len(self.images), channels, preprocessingProperties[0][0], preprocessingProperties[0][1]))
        self.audio = dataX[:, 0]
        self.audio = numpy.array([au for au in self.audio]).reshape((len(self.audio), self.audio[0].shape[0], 1))

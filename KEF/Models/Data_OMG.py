# -*- coding: utf-8 -*-
import numpy


class Data:
    dataX = None
    dataY = None
    labelDictionary = None

    def __init__(self, dataX=None, dataYAV=None, dataYEmotion=None, labelDictionary=None):
        self.dataX = dataX
        self.dataY = dataYAV
        self.dataYEmotion = dataYEmotion
        self.labelDictionary = labelDictionary

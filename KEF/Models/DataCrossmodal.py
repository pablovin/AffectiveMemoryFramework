# -*- coding: utf-8 -*-
import numpy


class DataCrossmodal:
    dataXAudio = None
    dataXVideo = None
    dataY = None
    labelDictionary = None

    def __init__(self, dataXAudio=None, dataXVideo=None, dataY=None, labelDictionary=None):
        self.dataXAudio = dataXAudio
        self.dataXVideo = dataXVideo
        self.dataY = dataY
        self.labelDictionary = labelDictionary

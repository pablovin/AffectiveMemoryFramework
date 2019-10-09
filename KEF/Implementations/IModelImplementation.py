# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty


class IModelImplementation():
    
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def modelName(self):        
        pass
    
    
    @abstractproperty
    def model(self):        
        pass

    @abstractmethod
    def buildModel(self, dataPoints):
        pass
          
    @abstractmethod
    def train(self, dataPoints):
        pass
    
    
    @abstractmethod
    def evaluate(self, dataPoints):
        pass
        
    @abstractmethod
    def classify(self, dataPoints):
        pass
    
    @abstractmethod
    def save(self, dataPoints):
        pass
    
    
    @abstractmethod
    def load(self, dataPoints):
        pass
        
    
    
    
    
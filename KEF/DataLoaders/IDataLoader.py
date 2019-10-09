# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty


class IDataLoader():
    
    __metaclass__ = ABCMeta
    
    
    @abstractproperty          
    def dataTrain(self):        
        pass
    
    @abstractproperty          
    def dataValidation(self):        
        pass
                      
    @abstractproperty          
    def dataTest(self):        
        pass
              

    @abstractproperty
    def preProcessingProperties(self):        
        pass
    
    @abstractproperty
    def logManager(self):        
        pass

              
    @abstractmethod
    def orderClassesFolder(self, folder):
        pass

    @abstractmethod
    def orderDataFolder(self, folder):
        pass        
        
    @abstractmethod
    def loadTrainData(self, folder):
        pass
    
    @abstractmethod
    def loadTestData(self, folder):
        pass

    @abstractmethod
    def loadValidationData(self, folder):
        pass
    
    @abstractmethod
    def loadTrainTestValidationData(self, folder, percentage):
        pass    
    
    @abstractmethod
    def loadNFoldValidationData(self, folder, NFold):
        pass    

    @abstractmethod
    def preProcess(self, data, parameters):
        pass    
            
    @abstractmethod
    def saveData(self, folder):
        pass
        
    @abstractmethod
    def shuffleData(self, folder):
        pass

    @abstractmethod
    def __init__(self, logManager, preProcessingProperties=None):
            pass
    
    
    
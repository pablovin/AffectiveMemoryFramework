# -*- coding: utf-8 -*-
import datetime
import sys

class LogManager:
    """Log Manager Class
    
    This class manager the log function of the framework. Here the log is created, and written if necessary.
    
    
    Attributes:
        logDiretory (String): This variable keeps the directory which the log file is saved.
    
    
    Author: Pablo Barros
    Created on: 02.02.2017
    Last Update: 20.02.2017
    
    Todo:
        * Create functions to log images and graphs as well.
    """    
    
    @property
    def logDirectory(self):        
        return self._logDirectory        

    @property
    def verbose(self):        
        return self._verbose     

        
        
    def __init__(self, logDirectory=None, verbose=False):
        
        #sys.stdout = open(logDirectory+"2.txt",'w')
        
        """
        Constructor function, which basically verifies if the logdirectory is correct,
        and if so, or creates or loads the log file.
        
        Args:
            logDirectory (String): the directory where the log is / will be is saved
            verbose(Boolean): Indicates if the log will also be printed in the console
            
        Raises:
            
            Exception: if the logDirectory is invalid.
        
        """
        
        try:
            self.isLogDirectoryValid(logDirectory)
        except:
            raise Exception("Log file not found!")       
            
        else:
            self._logDirectory = logDirectory
            
        self._verbose = verbose


    def isLogDirectoryValid(self, logDirectory):
        """
            Function that verifies if the log directory is valid and is an openable document.
            
            Args:
                logDirectory (String): the directory where the log is / will be is saved
                
            Raises:
                
                Exception: if the logDirectory is invalid.
            
            Returns:
                True if succesfull, raises the exception otherwise.
                
        """
        try:
            open(logDirectory,"a")
        except:
            raise Exception("Log file not found!")                
        return True
                     
                     
                     
    def write(self, message):
        """
            Function that writes messages in the log.
            
            Args:
                message (String): The message which will be written in the log.
                
            Raises:
                
                Exception: if the logDirectory is invalid.
                    
        """
        
        try:
            logFile = open(self.logDirectory,"a")
        except:
            raise Exception("Log file not found!")       
            
        else:
            logFile.write(str(datetime.datetime.now()).replace(" ", "_")+"-"+str(message)+"\n")                
            logFile.close
            
            if self.verbose:
                print str(datetime.datetime.now()).replace(" ", "_")+"-"+str(message)
            
        
    def newLogSession(self, sessionName):
        """
            Function that writes a new session in the Log.
            
            Args:
                sessionName (String): The name of the new session
                
            Raises:
                
                Exception: if the logDirectory is invalid.
                    
        """
        
        try:
            logFile = open(self.logDirectory,"a")
        except:
            raise Exception("Log file not found! Looked at:", self.logDirectory)       
            
        else:
            logFile.write("-----------------------------------------------------------------------------------------------------\n")    
            logFile.write(str(sessionName+"\n"))    
            logFile.write("-----------------------------------------------------------------------------------------------------\n")    
            logFile.close       
            
            if self.verbose:
                print ("-----------------------------------------------------------------------------------------------------\n")    
                print str(sessionName)
                print ("-----------------------------------------------------------------------------------------------------\n")    
        

    def endLogSession(self):
        """
            Function that writes the end of a session in the Log.
            
            Args:
                sessionName (String): The name of the new session
                
            Raises:
                
                Exception: if the logDirectory is invalid.
                    
        """
        
        try:
            logFile = open(self.logDirectory,"a")
        except:
            raise Exception("Log file not found! Looked at:", self.logDirectory)       
            
        else:
            logFile.write("-----------------------------------------------------------------------------------------------------\n")                
            logFile.write("-----------------------------------------------------------------------------------------------------\n")    
            logFile.close       
            
            if self.verbose:
                print ("-----------------------------------------------------------------------------------------------------\n")                    
                print ("-----------------------------------------------------------------------------------------------------\n")  
    

        
        
        
                
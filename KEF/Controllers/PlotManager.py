# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy
from keras.utils import plot_model

import pylab as P


class PlotManager():
    


    @property
    def plotsDirectory(self):
        return self._plotsDirectory
        
        
    def __init__(self, plotDirectory):
        
        self._plotsDirectory = plotDirectory

    def createListPlot(self, lists,names):

        for l in range(len(lists)):
            plt.plot(lists[l])
            plt.title(names[l])

            plt.ylabel(names[l])
            plt.xlabel('Iteration')
            plt.savefig(self.plotsDirectory + "/" + names[l] + ".png")
            plt.clf()



    def createCategoricalHistogram(self, dataPoints,dataName):

        dataClasses = []
        for y in dataPoints.dataY:
            dataClasses.append(numpy.argmax(y))

        dataClasses = numpy.array(dataClasses)

        n, bins, patches = plt.hist(dataClasses, len(dataPoints.dataY[0]), facecolor='green', alpha=0.75)
        P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)


        plt.savefig(self.plotsDirectory + "/" + dataName + "Histogram" + ".png")
        plt.clf()

    def createDataArousalValenceHistogram(self,dataPoints,dataName):

        arousals = []
        valences = []
        #print "Shape:", numpy.shape(dataPoints.dataY)
        #print dataPoints.dataY[0]
        for y in dataPoints.dataY:

            arousals.append(y[0])
            valences.append(y[1])

        arousals = numpy.array(arousals)
        valences = numpy.array(valences)

        n, bins, patches = plt.hist(arousals, 20, facecolor='green', alpha=0.75)
        P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

        plt.savefig(self.plotsDirectory + "/" + dataName + "arousal" + ".png")
        plt.clf()

        n, bins, patches = plt.hist(valences, 20, facecolor='green', alpha=0.75)
        P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

        plt.savefig(self.plotsDirectory + "/" + dataName + "valence" + ".png")
        plt.clf()

        plt.scatter(arousals, valences, 0.3)
        # draw a default hline at y=1 that spans the xrange
        l = plt.axhline(y=0)
        l = plt.axvline(x=0)
        plt.axis([-1, 1, -1 , 1])
        plt.xlabel("Arousal")
        plt.ylabel("Valence")

        plt.savefig(self.plotsDirectory+"/"+dataName+"Arousal_Valence"+".png")
        plt.clf()

    def  creatModelPlot(self, model, modelName=""):
        print "Creating Plot at: " + str(self.plotsDirectory) + "/" + str(modelName) + "_plot.png"
        plot_model(model,to_file=self.plotsDirectory+"/"+modelName + "_plot.png" ,show_layer_names=True, show_shapes=True)

        
    def createTrainingPlot(self, trainingHistory, modelName=""):

        print "Creating Training Plot"
        
        metrics = trainingHistory.history.keys()

        for m in metrics:
            if m.startswith('lr'):
                metrics.remove(m)

        for i in range(len(metrics)):            
            
            if not "val" in metrics[i]:
                #print "Models:"+ metrics[i]+" - "+ str(trainingHistory.history[metrics[i]])
                #print "Models:val_"+ metrics[i]+" - "+ str(trainingHistory.history["val_"+metrics[i]])
                #print "-"
                
                plt.plot(trainingHistory.history[metrics[i]])
                plt.plot(trainingHistory.history["val_"+metrics[i]])
                
                
                plt.title("Model's " + metrics[i])
                
                plt.ylabel(metrics[i])
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper left')
                #print "Saving Plot:", self.plotsDirectory+"/"+modelName+metrics[i]+".png"
                plt.savefig(self.plotsDirectory+"/"+modelName+metrics[i]+".png")
                plt.clf()

    def plotLoss(self,loss_history):
        plt.gcf().clear()
        plt.figure(1)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.plot(loss_history, c='b')
        plt.savefig("loss.png")

    def plotAcc(self,acc_history):
        plt.gcf().clear()
        plt.figure(1)
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.plot(acc_history, c='b')
        plt.savefig("acc.png")


    def plotOutput(self,faceX, faceY):
        plt.gcf().clear()
        plt.figure(1)
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.plot(faceX, c='b')
        plt.plot(numpy.tranpose(faceY), c='r')
        plt.savefig("face_plot.png")
            
    def plotReward(self,avg_reward):
        plt.gcf().clear()
        plt.figure(1)
        plt.ylabel('Avg. Reward ')
        plt.xlabel('Episodes x 1000')
        plt.plot(avg_reward, c='b')
        plt.savefig(self.plotsDirectory+"/avg_reward.png")

        
        
        
        
        

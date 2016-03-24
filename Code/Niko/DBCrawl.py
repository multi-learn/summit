#!/usr/bin/env python

""" Library: Functions to generate databases of images """

# Import built-in modules
import os                               # for iteration throug directories

# Import 3rd party modules
import pandas as pd                     # for Series and DataFrames

# Import own modules

# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Prototype"           # Production, Development, Prototype
__date__	= 2016-03-25


# ### Function to iterate through given directory and return images paths and classLabels
def imgCrawl(path, sClassLabels, nameDB, nbClasses): 
        df = pd.DataFrame()
        
        i = 0

        for subdir, dirs, files in os.walk(path): # loop through subdirectories
                # sort dirs and files
    		dirs.sort(key=lambda y: y.lower())
    		files.sort(key=lambda y: y.lower())
                
                # if the amount of classes to be evalauated is limited
                if(i>=nbClasses):
                        break
                        
                # loop througe files and create df        
		for file in files:
                        pathOfFile = os.path.join(subdir, file) #path of file
                        head, classLabel = os.path.split(os.path.split(pathOfFile)[0]) # get directoryname of file as classLabel
            
                        # assign integer label for dataframe
                        classLabel = sClassLabels[sClassLabels == classLabel].index[0]
                        df = df.append({'classLabel': classLabel, 'pathOfFile': pathOfFile}, ignore_index=True) 
                i+=1
            
        # DatabaseName: if name is given use it, if not will use name of the Folder of DB
        if(nameDB==""):
                nameDB = os.path.basename(os.path.normpath(path))
    
        # return ImageDB
        return (df,nameDB)
	
# ### Function to determine Class-Labels with Integer representation
# function to determine Class-labels and return Series
def getClassLabels(path, nbClasses):
        data = os.listdir(path) # listdir returns all subdirectories
        data.sort(key=lambda y: y.lower())
	
        if(len(data)>nbClasses):
                data = data[0:nbClasses]
                index = range(0,nbClasses)
        else:
                index = range(0,len(data))
    
        # return Class-Labels
        return pd.Series(data,index)
# Imports

import os as os        # for iteration throug directories
import pandas as pd # for Series and DataFrames
import cv2          # for OpenCV 
import datetime     # for TimeStamp in CSVFile
from scipy.cluster.vq import * # for Clustering http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
import numpy as np  # for arrays
import time       # for time calculations
from feature_extraction_try import imgCrawl, getClassLabels

#in : npImages, color

# In order to calculate HOG, we will use a bag of word approach : cf SURF function, well documented. 

def imageSequencing(npImages, CELL_DIMENSION):

  blocksList=[]
  for i in range(len(npImages)):
    image = cv2.imread(npImages[0][1])
    resizedImage = reSize(image, CELL_DIMENSION)
    height, width, channels = resizedImage.shape
    blocksList.append(np.array([resizedImage[j*CELL_DIMENSION:j*CELL_DIMENSION+CELL_DIMENSION-1, i*CELL_DIMENSION:i*CELL_DIMENSION+CELL_DIMENSION-1, :] for i in range(width/CELL_DIMENSION) for j in range(height/CELL_DIMENSION)]))
  return np.array(blocksList)  

def reSize(image, CELL_DIMENSION):
  height, width, channels = image.shape
  if height%5==0 and width%5==0:
    resizedImage = image
  elif width%5==0:
    missingPixels = 5-height%5
    resizedImage = cv2.copyMakeBorder(image,0,missingPixels,0,0,cv2.BORDER_REPLICATE)
  elif height%5==0:
    missingPixels = 5-width%5
    resizedImage = cv2.copyMakeBorder(image,0,0,0,missingPixels,cv2.BORDER_REPLICATE)
  else:
    missingWidthPixels = 5-width%5
    missingHeightPixels = 5-height%5
    resizedImage = cv2.copyMakeBorder(image,0,missingHeightPixels,0,missingWidthPixels,cv2.BORDER_REPLICATE)
  # height, width, channels = resizedImage.shape
  # if height%5==0 and width%5==0:
  #   print ("My job has been done")
  return resizedImage

start = time.time()
path ='../../03-jeux-de-donnees/101_ObjectCategories'
testNpImages = [ [1,'testImage.jpg'] ]
print testNpImages[0][1]
print "Fetching Images in " + path

# get dictionary to link classLabels Text to Integers
# sClassLabels = getClassLabels(path)

# Get all path from all images inclusive classLabel as Integer
# dfImages = imgCrawl(path, sClassLabels)
# npImages = dfImages.values
middle = time.time()
print "Extracted images in " + str(middle-start)
print "Sequencing Images ..."
sequencedCorpus = imageSequencing(testNpImages, 5)
end = time.time()
print "Sequenced images in " + str(end-middle)
print sequencedCorpus.shape
print sequencedCorpus[0][0]





# def even(difference):
#   return not(difference % 2)

# def findmaxDim(npImages):
  
#   max_height = 0
#   max_width = 0
  
#   for npImage in npImages:
#     height, width, channels = cv2.imread(npImage[1]).shape
#     if height > max_height:
#       max_height=height
#     if width > max_width:
#       max_width=width
#   return [max_height, max_width]

# def resizeImage(image, height, width):
#   ratio = float(8000000)/(width*height)
#   smallImage = cv2.resize(image, (0,0), fx=ratio, fy=ratio)
#   return smallImage 

# def enlarge(image, maxDimension, color):

#   height, width, channels = image.shape
#   [height_difference, width_difference] = np.array(maxDimension) - np.array([height, width])
#   print(height_difference, width_difference)
#   if even(height_difference) and even(width_difference):
#     treatedImage = cv2.copyMakeBorder(image, height_difference/2, height_difference/2, width_difference/2, width_difference/2, cv2.BORDER_CONSTANT, value=color)
    
#   elif even(height_difference):
#     treatedImage = cv2.copyMakeBorder(image, height_difference/2, height_difference/2, width_difference/2+1, width_difference/2, cv2.BORDER_CONSTANT, value=color)
    
#   elif even(width_difference):
#     treatedImage = cv2.copyMakeBorder(image, height_difference/2+1, height_difference/2, width_difference/2, width_difference/2, cv2.BORDER_CONSTANT, value=color)
    
#   else:
#     treatedImage = cv2.copyMakeBorder(image, height_difference/2+1, height_difference/2, width_difference/2+1, width_difference/2, cv2.BORDER_CONSTANT, value=color)
    
#   return treatedImage

# def calcHog(npImages, color, maxDimension):

#   list_hog = []
#   hog = cv2.HOGDescriptor()

#   # poulet = preTreat(cv2.imread(npImages[0][1]), maxDimension, color)
#   for npImage in npImages:
#     image = cv2.imread(npImage[1])
#     height, width, channels = image.shape

#     if height * width > 8000000:
#       g = hog.compute(resizeImage(image, height, width))
#     else:
#       g = hog.compute(enlarge(image, maxDimension, color))
#     print g.shape
#   # list_hog = [hog.compute(cv2.imread(npImage[1])) for npImage in npImages]
#   return list_hog

# color=[0,0,0]
# path ='../../03-jeux-de-donnees/101_ObjectCategories'

# # get dictionary to link classLabels Text to Integers
# sClassLabels = getClassLabels(path)

# # Get all path from all images inclusive classLabel as Integer
# dfImages = imgCrawl(path, sClassLabels)

# npImages = dfImages.values
# maxDimension = findmaxDim(npImages)
# list_hog = calcHog(npImages, color, maxDimension)
# print len(list_hog)

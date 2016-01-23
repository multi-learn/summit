# Imports

import os as os        # for iteration throug directories
import pandas as pd # for Series and DataFrames
import cv2          # for OpenCV 
import datetime     # for TimeStamp in CSVFile
from scipy.cluster.vq import * # for Clustering http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
import numpy as np  # for arrays
import time       # for time calculations
from feature_extraction_try import imgCrawl, getClassLabels
from skimage.feature import hog
from sklearn import cluster.MiniBatchKMeans

#in : npImages, color

# In order to calculate HOG, we will use a bag of word approach : cf SURF function, well documented. 

def imageSequencing(npImages, CELL_DIMENSION):

  blocks=[]
  for k in range(len(npImages)):
    image = cv2.imread(npImages[k][1])
    resizedImage = reSize(image, CELL_DIMENSION)
    height, width, channels = resizedImage.shape
    blocks.append(\
      np.array([\
        resizedImage[\
          j*CELL_DIMENSION:j*CELL_DIMENSION+CELL_DIMENSION,\
          i*CELL_DIMENSION:i*CELL_DIMENSION+CELL_DIMENSION] \
        for i in range(width/CELL_DIMENSION) \
        for j in range(height/CELL_DIMENSION)\
      ])\
    )
  return np.array(blocks)  

def reSize(image, CELL_DIMENSION):
  height, width, channels = image.shape
  if height%CELL_DIMENSION==0 and width%CELL_DIMENSION==0:
    resizedImage = image
  elif width%CELL_DIMENSION==0:
    missingPixels = CELL_DIMENSION-height%CELL_DIMENSION
    resizedImage = cv2.copyMakeBorder(image,0,missingPixels,0,0,cv2.BORDER_REPLICATE)
  elif height%CELL_DIMENSION==0:
    missingPixels = CELL_DIMENSION-width%CELL_DIMENSION
    resizedImage = cv2.copyMakeBorder(image,0,0,0,missingPixels,cv2.BORDER_REPLICATE)
  else:
    missingWidthPixels = CELL_DIMENSION-width%CELL_DIMENSION
    missingHeightPixels = CELL_DIMENSION-height%CELL_DIMENSION
    resizedImage = cv2.copyMakeBorder(image,0,missingHeightPixels,0,missingWidthPixels,cv2.BORDER_REPLICATE)
  return resizedImage

def hogAllBlocks(blocks):
  print blocks[0][0].shape
  gradients = np.array([np.array([hog(cv2.cvtColor(block, cv2.COLOR_BGR2GRAY), orientations=8, pixels_per_cell=(5,5), cells_per_block=(1,1)) for block in image]) for image in blocks])
  print gradients.shape
  return gradients

def clusterGradtiens(hogs, NB_CLUSTERS, maxIter):
  sizes = np.array([len(hog) for hog in hogs])
  nbImages =  len(hogs)
  flattenedHogs = hogs.flatten() 
  miniBatchKMeans = MiniBatchKMeans(n_clusters=NB_CLUSTERS, max_iter=maxIter, compute_labels=True)
  hogsLabels = miniBatchKMeans.fit_predict(flattenedHogs, y=None)
  return hogsLabels, sizes

def makeHistograms(labels, NB_CLUSTERS, sizes):
  indiceInLabels = 0
  hogs = []
  for image in sizes:
    histogram = np.zeros(NB_CLUSTERS)
    for i in range(image):
      indiceInLabels+=i 
      histogram[labels[indiceInLabels]]++
    hogs.append(histogram)
  return hogs

# Main for testing
if __name__ == '__main__':


  start = time.time()
  path ='../../03-jeux-de-donnees/101_ObjectCategories'
  testNpImages = [ [1,'testImage.jpg'] ]
  NB_CLUSTERS = 12
  print testNpImages[0][1]
  print "Fetching Images in " + path

  # get dictionary to link classLabels Text to Integers
  # sClassLabels = getClassLabels(path)

  # Get all path from all images inclusive classLabel as Integer
  # dfImages = imgCrawl(path, sClassLabels)
  # npImages = dfImages.values
  extractedTime = time.time()
  print "Extracted images in " + str(int(extractedTime-start)) +'sec'
  print "Sequencing Images ..."
  blocks = imageSequencing(testNpImages, 5)
  sequencedTime = time.time()
  print "Sequenced images in " + str(int(sequencedTime-extractedTime)) +'sec'
  print "Computing gradient on each block ..."
  gradients = hogAllBlocks(blocks)
  hogedTime = time.time()
  print "Computed gradients in " + str(int(hogedTime - sequencedTime)) + 'sec'
  print "Clustering gradients ..."
  gradientLabels, sizes = clusterGradients(gradients, NB_CLUSTERS, maxIter)
  clusteredItme = time.time()
  print "Clustered gradients in " + str(int(hogedTime - sequencedTime)) + 'sec'
  print "Computing histograms ..."
  histograms = makeHistogram(labels, NB_CLUSTERS, sizes)
  end = time.time()
  print "Computed histograms in " + str(int(end - hogedTime)) + 'sec'
  print "Total time : " str(int(end-start)) + 'sec'




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

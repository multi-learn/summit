# Imports

import os as os        # for iteration throug directories
import pandas as pd # for Series and DataFrames
import cv2          # for OpenCV 
import datetime     # for TimeStamp in CSVFile
import numpy as np  # for arrays
import time       # for time calculations
from feature_extraction_try import imgCrawl, getClassLabels
from skimage.feature import hog
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import Pool, Queue, Process #for parallelization https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool


# In order to calculate HOG, we will use a bag of word approach : cf SURF function, well documented. 

def reSize(image, CELL_DIMENSION):
  height, width, channels = image.shape
  
  if height%CELL_DIMENSION==0 and width%CELL_DIMENSION==0:
    resizedImage = image
  
  elif width%CELL_DIMENSION==0:
    missingPixels = CELL_DIMENSION-height%CELL_DIMENSION
    resizedImage = cv2.copyMakeBorder(image,0,missingPixels,0,\
                                      0,cv2.BORDER_REPLICATE)
  
  elif height%CELL_DIMENSION==0:
    missingPixels = CELL_DIMENSION-width%CELL_DIMENSION
    resizedImage = cv2.copyMakeBorder(image,0,0,0,missingPixels,\
                                      cv2.BORDER_REPLICATE)
  
  else:
    missingWidthPixels = CELL_DIMENSION-width%CELL_DIMENSION
    missingHeightPixels = CELL_DIMENSION-height%CELL_DIMENSION
    resizedImage = cv2.copyMakeBorder(image,0,missingHeightPixels,0,\
                                      missingWidthPixels,cv2.BORDER_REPLICATE)
  return resizedImage

class NpImage:
  def __init__(self, npImage):
    self.npImage=npImage
    self.cells = None

  def imageSequencing(self, CELL_DIMENSION):
    image = cv2.imread(self.npImage[1])
    resizedImage = reSize(image, CELL_DIMENSION)
    height, width, channels = resizedImage.shape

    self.cells = \
      np.array([\
        resizedImage[\
          j*CELL_DIMENSION:j*CELL_DIMENSION+CELL_DIMENSION,\
          i*CELL_DIMENSION:i*CELL_DIMENSION+CELL_DIMENSION] \
        for i in range(width/CELL_DIMENSION) \
        for j in range(height/CELL_DIMENSION)\
      ])
    print self.cells.shape
  
  def computeLocalHistogramsImage(self, NB_ORIENTATIONS, CELL_DIMENSION, nbCells):
    localHistograms = ["histo" for j in range(nbCells)]
    for j in range(nbCells):
      localHistograms[j] = hog(cv2.cvtColor(self.cells[j], cv2.COLOR_BGR2GRAY), \
                              orientations=NB_ORIENTATIONS, \
                              pixels_per_cell=(CELL_DIMENSION,\
                                              CELL_DIMENSION),\
                              cells_per_block=(1,1)) 
    self.localHOGs = localHistograms




def corpusSequencing(npImages, CELL_DIMENSION):
  nbImages = len(npImages)
  sequencing = ["cells" for i in range(nbImages)]
  images = [NpImage(npImages[i]) for i in range(nbImages)]
  for i in range(10):
    sequencing[i] = Process(target=NpImage.imageSequencing, args=(images[i], CELL_DIMENSION))
    sequencing[i].start()
  for i in range(10):  
    sequencing[i].join()
    print images[i].cells

  corpusCells = np.array([image.cells for image in images])
  print corpusCells.shape
  return corpusCells, images


def computeLocalHistograms(images, NB_ORIENTATIONS, CELL_DIMENSION):
  nbImages = len(images)
  nbCells = [len(image.cells) for image in images]
  localHistogramscomputing = [["histo" for j in range(nbCells[i])] for i in range(nbImages)]
  
  for i in range(nbImages):
    localHistogramscomputing[i] = Process(target=NpImage.computeLocalHistogramsImage, args=(images[i], NB_ORIENTATIONS, CELL_DIMENSION, nbCells[i]))
    localHistogramscomputing[i].start()
  for i in range(nbImages):
    localHistogramscomputing[i].join()
  
  localHistograms = [image.localHOGs for image in images]
  return localHistograms


def clusterGradients(localHistograms, NB_CLUSTERS, MAXITER):
  sizes = np.array([len(localHistogram) for localHistogram in localHistograms])
  nbImages =  len(localHistograms)
  flattenedHogs = np.array([cell for image in localHistograms for cell in image])
  miniBatchKMeans = MiniBatchKMeans(n_clusters=NB_CLUSTERS, max_iter=MAXITER, \
                    compute_labels=True)
  localHistogramLabels = miniBatchKMeans.fit_predict(flattenedHogs)
  return localHistogramLabels, sizes


def makeHistograms(labels, NB_CLUSTERS, sizes):
  indiceInLabels = 0
  hogs = []
  for image in sizes:
    histogram = np.zeros(NB_CLUSTERS)
    for i in range(image):
      histogram[labels[indiceInLabels+i]] += 1.0
    hogs.append(histogram/image)
    indiceInLabels+=i 
  return np.array(hogs)


def extractHOGFeature(npImages, CELL_DIMENSION, NB_ORIENTATIONS, \
                      NB_CLUSTERS, MAXITER):
  cells = corpusSequencing(npImages, CELL_DIMENSION)
  localHistograms = computeLocalHistograms(cells)
  localHistogramLabels, sizes = clusterGradients(localHistograms, \
                                                NB_CLUSTERS, MAXITER)
  hogs = makeHistograms(localHistogramLabels, NB_CLUSTERS, sizes)
  return hogs


# Main for testing
if __name__ == '__main__':


  start = time.time()
  # path='/home/doob/Dropbox/Marseille/OMIS-Projet/03-jeux-de-donnees/101_ObjectCategories'
  path ='/donnees/bbauvin/101_ObjectCategories'
  testNpImages = [ [1,'testImage.jpg'], [1,'testImage.jpg'] ]
  CELL_DIMENSION = 5
  NB_ORIENTATIONS = 8
  NB_CLUSTERS = 12
  MAXITER = 100

  print "Fetching Images in " + path
  # get dictionary to link classLabels Text to Integers
  sClassLabels = getClassLabels(path)
  # # Get all path from all images inclusive classLabel as Integer
  dfImages = imgCrawl(path, sClassLabels)
  npImages = dfImages.values
  extractedTime = time.time()
  print "Extracted images in " + str(extractedTime-start) +'sec'
  print "Sequencing Images ..."
  cells, images = corpusSequencing(npImages, CELL_DIMENSION)
  sequencedTime = time.time()
  print "Sequenced images in " + str(sequencedTime-extractedTime) +'sec'
  print "Computing gradient on each block ..."
  gradients = computeLocalHistograms(images, NB_ORIENTATIONS, CELL_DIMENSION)
  hogedTime = time.time()
  print "Computed gradients in " + str(hogedTime - sequencedTime) + 'sec'
  print "Clustering gradients ..."
  gradientLabels, sizes = clusterGradients(gradients, NB_CLUSTERS, MAXITER)
  clusteredItme = time.time()
  print "Clustered gradients in " + str(hogedTime - sequencedTime) + 'sec'
  print "Computing histograms ..."
  histograms = makeHistograms(gradientLabels, NB_CLUSTERS, sizes)
  end = time.time()
  print "Computed histograms in " + str(int(end - hogedTime)) + 'sec'
  print "Histogram shape : " +str(histograms.shape)
  print "Total time : " + str(end-start) + 'sec'
  # #hogs = extractHOGFeature(testNpImages, CELL_DIMENSION, \
  # #                         NB_ORIENTATIONS, NB_CLUSTERS, MAXITER)
  


# # Imports

# import os as os        # for iteration throug directories
# import pandas as pd # for Series and DataFrames
# import cv2          # for OpenCV 
# import datetime     # for TimeStamp in CSVFile
# import numpy as np  # for arrays
# import time       # for time calculations
# from feature_extraction_try import imgCrawl, getClassLabels
# from skimage.feature import hog
# from sklearn.cluster import MiniBatchKMeans
# from multiprocessing import Process #for parallelization

# def reSize(image, CELL_DIMENSION):
#   height, width, channels = image.shape
  
#   if height%CELL_DIMENSION==0 and width%CELL_DIMENSION==0:
#     resizedImage = image
  
#   elif width%CELL_DIMENSION==0:
#     missingPixels = CELL_DIMENSION-height%CELL_DIMENSION
#     resizedImage = cv2.copyMakeBorder(image,0,missingPixels,0,\
#                                       0,cv2.BORDER_REPLICATE)
  
#   elif height%CELL_DIMENSION==0:
#     missingPixels = CELL_DIMENSION-width%CELL_DIMENSION
#     resizedImage = cv2.copyMakeBorder(image,0,0,0,missingPixels,\
#                                       cv2.BORDER_REPLICATE)
  
#   else:
#     missingWidthPixels = CELL_DIMENSION-width%CELL_DIMENSION
#     missingHeightPixels = CELL_DIMENSION-height%CELL_DIMENSION
#     resizedImage = cv2.copyMakeBorder(image,0,missingHeightPixels,0,\
#                                       missingWidthPixels,cv2.BORDER_REPLICATE)
#   return resizedImage


# class HOGComputer:
#   def __init__(self, CELL_DIMENSION, NB_ORIENTATIONS, \
#                       NB_CLUSTERS, MAXITER, npImages):
#     self.CELL_DIMENSION  = CELL_DIMENSION
#     self.NB_ORIENTATIONS = NB_ORIENTATIONS
#     self.NB_CLUSTERS = NB_CLUSTERS
#     self.MAXITER = MAXITER
#     self.npImages = npImages
#     self.cells = [[] for k in range(len(npImages))]
#     self.localHistograms = np.array([])
#     self.localHistogramLabels = np.array([])
#     self.sizes = np.array([])
#     self.hogs = np.array([])


#   def imageSequencing(self, imageIndice):
#     image = cv2.imread(self.npImages[imageIndice][1])
#     print(image.shape)
#     resizedImage = reSize(image, self.CELL_DIMENSION)
#     height, width, channels = resizedImage.shape
#     self.cells[imageIndice] = \
#     [\
#       resizedImage[\
#         j*self.CELL_DIMENSION:j*self.CELL_DIMENSION+self.CELL_DIMENSION,\
#         i*self.CELL_DIMENSION:i*self.CELL_DIMENSION+self.CELL_DIMENSION\
#       ] for i in range(width/self.CELL_DIMENSION) \
#       for j in range(height/self.CELL_DIMENSION)\
#     ]
#     print len(self.cells[imageIndice])


#   def corpusSequencing(self):
#     nbImages = len(self.npImages)
#     processes = ["process" for k in range(nbImages)]
    
#     for imageIndice in range(nbImages):
#       processes[imageIndice]=Process(target=HOGComputer.imageSequencing, args=(self, imageIndice))
#       processes[imageIndice].start()
#     for imageIndice in range(nbImages):
#       processes[imageIndice].join()
#     print len(self.cells[0])
      
# if __name__=='__main__':
#   testNpImages = [ [1,'testImage.jpg'], [1, 'testImage.jpg'] ]
#   hogComputer = HOGComputer(5,8,12,100,testNpImages)
#   hogComputer.corpusSequencing()
#   print len(hogComputer.cells)
#   print len(hogComputer.cells[0])
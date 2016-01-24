# coding: utf-8
import os as os        # for iteration throug directories
import pandas as pd # for Series and DataFrames
import cv2          # for OpenCV 
import datetime     # for TimeStamp in CSVFile
from scipy.cluster.vq import * # for Clustering http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
import numpy as np  # for arrays
import time 


# # Code to Extract ColorHistograms for Database

# #### Author: Nikolas Hülsmann
# #### Date: 2015-11-22

# ## Functions for Extract Data
# 
# ### Function to iterate through given directory and return images paths and classLabels

# In[31]:

def imgCrawl(path, sClassLabels): #path to 'highest' folder
    rootdir = path
    df = pd.DataFrame()
        
    for subdir, dirs, files in os.walk(rootdir): # loop through subdirectories
        for file in files:
            pathOfFile = os.path.join(subdir, file) #path of file
            head, classLabel = os.path.split(os.path.split(pathOfFile)[0]) # get directoryname of file as classLabel
            
            # assign integer label for dataframe
            classLabel = sClassLabels[sClassLabels == classLabel].index[0]
            df = df.append({'classLabel': classLabel, 'pathOfFile': pathOfFile}, ignore_index=True) 
            
    return df


# ### Function to determine Class-Labels with Integer representation

# In[32]:

# function to determine Class-labels and return Series
def getClassLabels(path):
    data = os.listdir(path) # listdir returns all subdirectories
    index = range(0,len(data))
    
    return pd.Series(data,index)


# ### Function to calculate the ColorHistogram for given Images 

# In[33]:

#### Calculate ColorHistograms for all images

### Points to improve: 
# - use HSV color spectrum
# - change function: parameter how many bins of ColorHistogramm (feature length)


# dfImages: Dataframe with paths to all images - use function imgCrawl
# numberOfBins_: Number of bins Histogram
def calcColorHisto(dfImages_, numberOfBins_):
    # Initialize function
    df = pd.DataFrame()
    npImages = dfImages_.values
    numberOfBins = numberOfBins_
    npColorHist = np.zeros((len(npImages), numberOfBins*3), "float32")
    i=0
    
    ## algo
    for images in npImages:
        image = cv2.imread(images[1])
        
        # Image Size for Normalization
        height, width, channels = image.shape
        img_size = height * width
        
        # Split into color chanels rgb
        chans = cv2.split(image)
        colors = ("b", "g", "r")
        
        histogram = []

        ########### Feature Color Histogram (cf. http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html)     # loop over the image channels
        for (chan, color) in zip(chans, colors):         
            
            # Calculate Color Histogram - 16 bins cf. paper (test with 64 has shown that die result is similair in score)
            # Seperates the intesity for each color from 0 to 256, and creates 16 bins of same size: 0-15, 16-31, .. , 241-256
            hist = cv2.calcHist([chan], [0], None, [numberOfBins], [0, 256])

            # to get raw values
            hist = hist[:,0]
            
            # Normalize to a Distrubution from 0 to 1 throug calculating for each color channel (red/blue/green): 
            #        (number of pixels in bin)/(pixel size of image)
            #hist[:] = [x / img_size for x in hist]
            hist[:] = [x / sum(hist) for x in hist]
            

            # Normalize with MinMax from 0 to 1 -> feature scaling
            #cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            histogram.extend(hist)

        # append features_colHist to df
        npColorHist[i] = histogram
        i = i+1
        #df = df.append({'ColHisto': features_colHist}, ignore_index=True) 
    
    return npColorHist


# ### Function to calculate Surf Histogram

# In[34]:

################# FEATURE SURF (cf. http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html#surf)
# API cf. http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html

#### Calculate Histogramm of SURF Descripteurs with Bag Of Words appraoch for all images

### Points to improve: 
# - use spatial histogram: http://www.di.ens.fr/willow/events/cvml2011/materials/practical-classification/
# - change function: parameter how many K clustes/feature length (in regard of overfitting)


# path to higehst folder
# dfImages: Dataframe with paths to all images - use function imgCrawl
# k: number of K-Cluster -> length of feature vector
def calcSurfHisto(dfImages_, k_):
    
    # Initialize function
    df = pd.DataFrame()
    npImages = dfImages_.values
    k = k_
    
    # List where all the descriptors are stored
    des_list = []
    
    #### Feature Detection and Description (Surf): 
    # Detect (localize) for each image the keypoints (Points of Interest in an image - Surf uses therefore like SIFT corners)
    # Pro: SIFT/SURF are scale and rotation invariant!
    for images in npImages:
        # Read image
        image = cv2.imread(images[1])
        
        # Method to detect keypoints (kp) and calculate the descripteurs (des) with one function call
        # Each image has different amount of kp, but each kp has a describteur of fixed length (128)
        kp, des = sift.detectAndCompute(image,None)
        des_list.append(des)
    
    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for descriptor in des_list[0:]:
        descriptors = np.vstack((descriptors, descriptor)) 
    
    #### Bag of Words Approach
    ### 1. Step: using K-means cluster to create dictionary/vocabulary/codebook:
    # Encoding is the quantization of the image kp/des that constitute the image to be classified. 
    # Basic encoding schemes work by first running K-means on the set of all des that you collect 
    # across multiple images.
    # This builds what is known a dictionary/vocabulary/codebook represented by the centroids obtained from the clustering.
    
    # Perform k-means clustering -> creates the words from all describteurs -> this is the (dic) dictionary/vocabulary/codebook
    # k: amount of different clusters to build! Will result in a feature length k
    dic, variance = kmeans(descriptors, k, 1) 
    
    ### 2. Step: encoding/coding/vector quantization(vq) to assign each descripteur the closest "visual word" from dictionary:
    # At the end of this process, you end up with K representative "visual words" (the centroid of each cluster after 
    # K means ends) of your image descripteurs. These "visual words" represent what is usually understood as your 
    # visual dictionary. Once you have these visual words, encoding is the process of assigning 
    # each descripteur within your image the "visual word" (nearest neighbor) in the dictionary.
    
    npSurfHist = np.zeros((len(npImages), k), "float32")
    for i in xrange(len(npImages)):
        # vq: (Encoding) Assign words from the dictionary to each descripteur
        words, distance = vq(des_list[i],dic)
        
        ### 3. Step: Pooling - calculate a histogram for each image
        # Pooling refers to the process of representing an image as a "bag of words". 
        # The word bag here is meant to convey that once you have encoded each descripteur with a word  (a number between 1 and K), 
        # you build a new representation (a bag) that discards the spatial relationship between the words that 
        # constitute your image.

        # This representation is often a histogram or a collection of spatially adjacent histograms of the desribteurs 
        # (i.e. histograms of values 1 to K) that together form your image. "Pooling" is thus the process of 
        # building a histogram of words (i.e. pooling ~ "sampling" words from the image to build a probability 
        # mass function of words)

        # To clarify, the purpose of pooling is two fold:
        #           By building a feature vector that is a histogram of words (as opposed to putting the full "sentence of words" 
        #           in the feature vector), your descriptor will be invariant to changes in "the ordering of words". 
        #           In computer vision this translates into invariance with respect to rotations and distortions of the image 
        #           and object, which is a desirable thing to have.

        #           If the dictionary is small compared to the length of the sentence, a histogram of words has less dimensions 
        #           than the original vector. Less dimensions makes learning (training) much easier.
        
        
        # Count the accuarance of each word (w) in image (i) to build histogram
        for w in words:
            npSurfHist[i][w] += 1
        
        #### 4. Step: Normalization of features vector (Can be changed to distribution like ColorHisto)
        # Frequency divided by amount of words (k)
        summe = sum(npSurfHist[i])
        for x in range(0,k):
            #npSurfHist[i][x] = npSurfHist[i][x]/k
            npSurfHist[i][x] = npSurfHist[i][x]/summe
        
        #stdSlr = StandardScaler().fit(npSurfHist)
        #npSurfHist = stdSlr.transform(npSurfHist)
    
    return npSurfHist


# ### SIFT Experimental - use SURF 

# In[35]:

# ########### Feature SIFT (Scale-invariant feature transform cf. http://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html#gsc.tab=0)
# # Api cf. http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html
# import cv2
# import numpy as np

# img = cv2.imread('../../03-jeux-de-donnees/101_ObjectCategories/airplanes/image_0306.jpg')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sift = cv2.SIFT(nfeatures=100)
# #sift = cv2.xfeatures2d.SIFT_create()

# # Detector which detects the Keypoints in the Image
# #kp = sift.detect(gray,None)

# # Just a visualization of the Keypoints in the Image
# #img=cv2.drawKeypoints(gray,kp)
# #cv2.imwrite('D:\Sift-test\sift_keypoints.jpg',img)

# # Another visualization with FLAG: draw a circle with size of keypoint and it will even show its orientation
# #img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# #cv2.imwrite('D:\Sift-test\sift_keypoints.jpg',img)

# # Method to compute the descripteurs after one has already detected the keypoints
# #kp,des = sift.compute(gray,kp)

# #sift = cv2.xfeatures2d.SIFT_create()
# #sift = cv2.SIFT()

# # Method to detect keypoints (kp) and calculate the descripteurs (des) with one function call
# kp, des = sift.detectAndCompute(gray,None)

# print (des.shape)


# ### Functions to export calculated Data to csv 

# In[36]:

#### Export Features to csv
def exportToCSV(pandasSorDF, filename):
    #filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-Feature"
    path = os.getcwdu() + "\\" + filename
    
    if os.path.isfile(path + ".csv"):
        for i in range(1,20):
            testFileName = filename  + "-" + str(i) + ".csv"
            if os.path.isfile(os.getcwdu() + "\\" +  testFileName)!=True:
                pandasSorDF.to_csv(testFileName)
                break

    else:
        pandasSorDF.to_csv(filename + ".csv")


# In[37]:

def exportNumpyToCSV(numpyArray, filename):
    #filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-Feature"
    path = os.getcwdu() + "\\" + filename
    
    if os.path.isfile(path + ".csv"):
        for i in range(1,20):
            testFileName = filename  + "-" + str(i) + ".csv"
            if os.path.isfile(os.getcwdu() + "\\" +  testFileName)!=True:
                np.savetxt(testFileName, numpyArray, delimiter=",")
                break

    else:
        np.savetxt(filename + ".csv", numpyArray, delimiter=",")


# ## Main Programm
# 

# In[38]:

# # Imports
# import os           # for iteration throug directories
# import pandas as pd # for Series and DataFrames
# import cv2          # for OpenCV 
# import datetime     # for TimeStamp in CSVFile
# from scipy.cluster.vq import * # for Clustering http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
# import numpy as np  # for arrays
# import time       # for time calculations


# # In[39]:

# start = time.time()

# # Determine the Database to extract features
# path ='../../03-jeux-de-donnees/101_ObjectCategories'

# # get dictionary to link classLabels Text to Integers
# sClassLabels = getClassLabels(path)

# # Get all path from all images inclusive classLabel as Integer
# dfImages = imgCrawl(path, sClassLabels)

# print dfImages.classLabel.shape

# fileNameClassLabels = datetime.datetime.now().strftime("%Y_%m_%d") + "-Caltech-ClassLabels"
# exportNumpyToCSV(dfImages.classLabel, fileNameClassLabels)

# fileNameClassLabels = datetime.datetime.now().strftime("%Y_%m_%d") + "-Caltech-ClassLabels-Description"
# #exportToCSV(sClassLabels, fileNameClassLabels)

# end = time.time()
# print "Time to extract all images: " + str(end - start)


# # In[ ]:

# start = time.time()

# # Calculate Color Histogramm wit 16 bins for each color -> feature length = 3 x 16 = 48
# npColorHistogram = calcColorHisto(dfImages, 16)

# print npColorHistogram.shape

# fileNameColorHis = datetime.datetime.now().strftime("%Y_%m_%d") + "-Caltech-Feature-ColorHistogram"
# #exportNumpyToCSV(npColorHistogram, fileNameColorHis)

# end = time.time()
# print "Time to calculate ColorHistogram: " + str(end - start)


# # In[ ]:

# start = time.time()

# # Calculate Surf Histogramm with K=100 Cluster
# npSurfHistogram = calcSurfHisto(dfImages, 5)

# print npSurfHistogram.shape

# fileNameSurfHis = datetime.datetime.now().strftime("%Y_%m_%d") + "-Caltech-Feature-SurfHistogram"
# #exportNumpyToCSV(npSurfHistogram, fileNameSurfHis)

# end = time.time()
# print "Time to calculate SurfHistogram: " + str(end - start)


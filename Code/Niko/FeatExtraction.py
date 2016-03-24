#!/usr/bin/env python

""" Library: Code to extract all Features from a Database """

# Import built-in modules
  
  
# Import 3rd party modules
import numpy as np                                                      # for numpy arrays  
import cv2                                                              # for OpenCV 
from scipy.cluster.vq import *                                          # for Clustering http://docs.scipy.org/doc/scipy/reference/cluster.vq.html\n",
import logging                                                          # To create Log-Files  

# Import own modules
from hog_extraction_parallelized import extractHOGFeatureParallel       #For HOG
from hog_extraction import extractHOGFeature                            #for HOG computation

# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Prototype"                                           # Production, Development, Prototype
__date__	= 2016-03-25


#### Calculate RGBColorHistograms for all images

# nameDB: Name of the current Image DB
# dfImages: Dataframe with paths to all images - use function imgCrawl
# numberOfBins_: Number of bins Histogram
# maxColorIntensity_ : Seperates the intesity for each color from 0 to maxColorIntensity
# boolNormMinMax_ : True -> use MinMax Norm from 0 to 1 ; False -> Distribution from 0 to 1
def calcRGBColorHisto(nameDB, dfImages, numberOfBins, maxColorIntensity, boolNormMinMax):
        # Initialize function
        npImages = dfImages.values
        npColorHist = np.zeros((len(npImages), numberOfBins*3), "float32")
        i=0 
	
        # Description of Feature
        if(boolNormMinMax==True):
                norm = "MinMax"
        else:
                norm = "Distr"
                param = "Bins_" +  str(int(numberOfBins)) + "-" + "MaxCI_" + str(maxColorIntensity) + "-" + "Norm_" + norm
                description = nameDB + "-RGB-" + param
	
    
        ## algo
        for images in npImages:
                image = cv2.imread(images[1])
        
                # Image Size for Normalization
                #height, width, channels = image.shape
                #img_size = height * width
                
                # Split into color chanels rgb
                chans = cv2.split(image)
                colors = ("b", "g", "r")
                
                histogram = []

                ########### Feature Color Histogram (cf. http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html)     # loop over the image channels
                for (chan, color) in zip(chans, colors):         
            
                        # Calculate Color Histogram - numberOfBins bins cf. paper (test with 64 has shown that die result is similair in score)
                        # Seperates the intesity for each color from 0 to maxColorIntensity

                        # Example: Bins=16 and colorintensity=[0,256] -> creates 16 bins of same size: 0-15, 16-31, .. , 241-256
           
                        hist = cv2.calcHist([chan], [0], None, [numberOfBins], [0, maxColorIntensity])

                        # to get raw values
                        hist = hist[:,0]
			
			# Check if hist has values
                        sumHist = sum(hist) # Calculates the pixels of each image (images of different resolutions)
			
                        if(sumHist==0):
                                logging.warning("WARNING NORMALIZATION: sumHIST is zero")
                                logging.warning("image: " + images[1] + "\n")
            
                        # Normalization
                        if(boolNormMinMax == False):			
                                # Normalize to a Distrubution from 0 to 1 throug calculating for each color channel (red/blue/green): 
                                # (number of pixels in bin)/(sum of pixels in histogram)
                                # hist[:] = [x / img_size for x in hist]
                                #sumHist = sum(hist)
                                if(sumHist>0):
                                        hist[:] = [x / sumHist for x in hist]
                                else:
                                        # Normalize with MinMax from 0 to 1 -> feature scaling
                                        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
                        histogram.extend(hist)

                # append features_colHist to npColorHist
                npColorHist[i] = histogram
                i = i+1

        return (description,npColorHist)


#### Calculate HSVColorHistograms for all images
### Points to improve: 
# is this feature really working?

# nameDB: Name of the current Image DB
# dfImages: Dataframe with paths to all images - use function imgCrawl
# histSize: Number of bins in Histogram (differenz for each channel H,S,V)
# boolNormMinMax_ : True -> use MinMax Norm from 0 to 1 ; False -> Distribution from 0 to 1
def calcHSVColorHisto(nameDB, dfImages, histSize_, boolNormMinMax):
        # Initialize function
        npImages = dfImages.values
        histSize = histSize_
        npColorHist = np.zeros((len(npImages), int(histSize[0]+histSize[1]+histSize[2])), "float32")
        i=0
	
	# Description of Feature
        if(boolNormMinMax==True):
                norm = "MinMax"
        else:
                norm = "Distr"
        
        param = "Bins_" +  str(histSize) + "-" + "Norm_" + norm
        param = param.replace(" ", "")
        description = nameDB + "-HSV-" + param
	

        h_ranges = [ 0, 180 ]
        s_ranges = [ 0, 256 ]
        v_ranges = [ 0, 256 ]

        ranges = []

        ranges.append(h_ranges)
        ranges.append(s_ranges)
        ranges.append(v_ranges)

        channels = [0, 1, 2]
    
        histogram = []
    
        ## algo
        for images in npImages:
                image = cv2.imread(images[1])
                hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                
                
                # Split into color chanels rgb
                chans = cv2.split(image)
                colors = ("H", "S", "V")
        
                histogram = []
        
                ########### Feature Color Histogram (cf. http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html)     # loop over the image channels
                for (chan, color, binsize, range_chan) in zip(chans, colors, histSize, ranges): 
                        hist = cv2.calcHist([chan], [0], None, [binsize], range_chan )
            
                        # to get raw values
                        hist = hist[:,0]
            
                        # Check if hist has values
                        sumHist = sum(hist) # Calculates the pixels of each image (images of different resolutions)
			
                        if(sumHist==0):
                                logging.warning("WARNING NORMALIZATION: sumHIST is zero")
                                logging.warning("image: " + images[1] + "\n")
            
                        # Normalization
                        if(boolNormMinMax == False):			
                                # Normalize to a Distrubution from 0 to 1 throug calculating for each color channel (H/S/V):
				#        (number of pixels in bin)/(sum of pixels in histogram)
				#hist[:] = [x / img_size for x in hist]
                                #sumHist = sum(hist)
                                if(sumHist>0):
                                        hist[:] = [x / sumHist for x in hist]
                        else:
				# Normalize with MinMax from 0 to 1 -> feature scaling
                                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                        
                        histogram.extend(hist)
        
        
                # append features_colHist to npColorHist
                npColorHist[i] = histogram
                i = i+1
        
        return (description,npColorHist)


#### Calculate the SIFT/SURF descripteurs for all images
# dfImages: Dataframe with paths to all images - use function imgCrawl
# boolSIFT: To determine which feature to use: True->SIFT , False->SURF
def calcSURFSIFTDescriptors(dfImages, boolSIFT):
        # Initialize function
        npImages = dfImages.values
    
        if(boolSIFT==True):
                feat = "SIFT:\t "
        else:
                feat = "SURF:\t "
        
        # List where all the descriptors are stored
        des_list = []
        
       
        logging.debug(feat + "Keypoints Calculation")
        #### Feature Detection and Description (Surf): 
        # Detect (localize) for each image the keypoints (Points of Interest in an image - Surf uses therefore like SIFT corners)
        # Pro: SIFT/SURF are scale and rotation invariant!
        
        # bool for Progressbar
        bool_Progress=True
        
        for images,i in zip(npImages,range(1,len(npImages)+1)):
                # Read image
                if(float(i)/float(len(npImages))>0.25 and bool_Progress==True):
                        logging.debug(feat + "25% of images processed (Keypoints)")
                        bool_Progress = False
                elif(float(i)/float(len(npImages))>0.5 and bool_Progress==False):
                        logging.debug(feat + "50% of images processed (Keypoints)")
                        bool_Progress = None
                elif(float(i)/float(len(npImages))>0.75 and bool_Progress==None):
                        logging.debug(feat + "75% of images processed (Keypoints)")
                        bool_Progress = NotImplemented

                image = cv2.imread(images[1], cv2.CV_LOAD_IMAGE_COLOR)
                # Method to detect keypoints (kp) and calculate the descripteurs (des) with one function call
                # Each image has different amount of kp, but each kp has a describteur of fixed length (128)
                if(boolSIFT==True):
                        det = cv2.SIFT()
                else:
                        det = cv2.SURF()
                        
                kp, des = det.detectAndCompute(image,None)
                
                if des is None:
                        logging.debug(feat + "No Keypoints found in: " + str(images[1]))
                        if(boolSIFT==True):
                                desNoKP=np.zeros(shape=(100,128)) 
                        else: 
                                desNoKP=np.zeros(shape=(100,64)) 
                        des_list.append(desNoKP)
    
                if des is not None:
                        des_list.append(des)
                        #print des.shape
                        #print type(des[0])
                        #print type(des[0][0])
    
        #print "SIFT/SURF: Descriptory vertically"
        # Stack all the descriptors vertically in a numpy array
        #descriptors = des_list[0]
        #for descriptor,i in zip(des_list[1:],range(1,len(des_list)+1)):
                #print "DescriptorNr:" + str(i) + " of:" + str(len(des_list))
                #descriptors = np.vstack((descriptors, descriptor)) 
                #print descriptors.shape
        
        size = 0
        
        for i in range(0,len(des_list)):
            size = size + len(des_list[i])
            #print "SIFT/SURF: Actuel length:" + str(size)
        
        #print feat + "Length of Descriptors: " + str(size)
        
        merker = 0
        
        if(boolSIFT==True):
                descriptors = np.zeros(shape=(size,128), dtype=np.float32)
        else:
                descriptors = np.zeros(shape=(size,64), dtype=np.float32)
        
        #logging.debug(feat + "Start filling descriptors")
        for i in range(0,len(des_list)):
                descriptors[merker:(merker+len(des_list[i]))] = des_list[i]
                merker = merker + len(des_list[i])
        
        #logging.debug(feat + "Shape of Descriptors: " + str(descriptors.shape))
       
        
        return (descriptors,des_list)

        
        
# ### Function to calculate Surf Histogram
################# FEATURE SURF (cf. http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html#surf)
# API cf. http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html

#### Calculate Histogramm of SURF Descripteurs with Bag Of Words appraoch for all images

### Points to improve: 
# - use spatial histogram: http://www.di.ens.fr/willow/events/cvml2011/materials/practical-classification/
# - find parameters: minHessian 1000
# - resize Image to 640*480?
# - grayscale Image
# - 

### Points not yet tested:
# - MinMax Normalization

# nameDB: Name of the current Image DB
# dfImages: Dataframe with paths to all images - use function imgCrawl
# k: number of K-Cluster -> length of feature vector
# boolNormMinMax_ : True -> use MinMax Norm from 0 to 1 ; False -> Distribution from 0 to 1
# descriptors: Descripteurs of keypoints see function: calcSURFSIFTDescriptors
# des_list: List is needed for Dictonary
# boolSIFT: To determine which feature to use: True->SIFT , False->SURF
def calcSURFSIFTHisto(nameDB, dfImages, cluster, boolNormMinMax, descriptors,des_list, boolSIFT):
        #print "Begin of SURFSIFTHisto"
        # Initialize function
        npImages = dfImages.values
	
	# Description of Feature
        if(boolNormMinMax==True):
                norm = "MinMax"
        else:
                norm = "Distr"
	
        param = "Cluster_" +  str(cluster) + "-" + "Norm_" + norm
        if(boolSIFT==True):
                feat="SIFT"
        else:
                feat="SURF"
        description = nameDB + "-" + feat + "-" + param
        
        #### Bag of Words Approach
        ### 1. Step: using K-means cluster to create dictionary/vocabulary/codebook:
        # Encoding is the quantization of the image kp/des that constitute the image to be classified. 
        # Basic encoding schemes work by first running K-means on the set of all des that you collect 
        # across multiple images.
        # This builds what is known a dictionary/vocabulary/codebook represented by the centroids obtained from the clustering.
    
        # Perform k-means clustering -> creates the words from all describteurs -> this is the (dic) dictionary/vocabulary/codebook
        # k: amount of different clusters to build! Will result in a feature length k
        logging.debug(feat + ":\t Calculation of Dictonary with " + str(int(cluster)) + " Clusters")
        dic, variance = kmeans(descriptors, int(cluster), 1) 
        
        ### 2. Step: encoding/coding/vector quantization(vq) to assign each descripteur the closest "visual word" from dictionary:
        # At the end of this process, you end up with cluster representative "visual words" (the centroid of each cluster after 
        # K means ends) of your image descripteurs. These "visual words" represent what is usually understood as your 
        # visual dictionary. Once you have these visual words, encoding is the process of assigning 
        # each descripteur within your image the "visual word" (nearest neighbor) in the dictionary.
        
        # bool for Progressbar
        bool_Progress=True
        
        logging.debug(feat + ":\t Assign words from Dictonary to each Image")
        npSurfHist = np.zeros((len(npImages), int(cluster)), dtype=np.float32)
        for i in xrange(len(npImages)):
                # vq: (Encoding) Assign words from the dictionary to each descripteur
                words, distance = vq(des_list[i],dic)
                
                if(float(i)/float(len(npImages))>0.25 and bool_Progress==True):
                        logging.debug(feat + ":\t 25% of images processed (Assignments)")
                        bool_Progress = False
                elif(float(i)/float(len(npImages))>0.5 and bool_Progress==False):
                        logging.debug(feat + ":\t 50% of images processed (Assignments)")
                        bool_Progress = None
                elif(float(i)/float(len(npImages))>0.75 and bool_Progress==None):
                        logging.debug(feat + ":\t 75% of images processed (Assignments)")
                        bool_Progress = NotImplemented
        
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
             
		#### 4. Step: Normalization of features vector
	 
		# Check if hist has values
                sumHist = sum(npSurfHist[i]) # Calculates the pixels of each image (images of different resolutions)
		
                if(sumHist==0):
                        logging.warning("WARNING NORMALIZATION: sumHIST is zero")
                        logging.warning( "image: " + images[1] + "\n")
		
		# Normalization
                if(boolNormMinMax == False):			
			# Frequency divided by amount of words (cluster)
                        if(sumHist>0):
                                for x in range(0,int(cluster)):
                                        npSurfHist[i][x] = npSurfHist[i][x]/sumHist #sumHist can be zero...change system
                else:
			# Normalize with MinMax from 0 to 1 -> feature scaling
                        for x in range(0,int(cluster)):
                                cv2.normalize(npSurfHist[i][x], npSurfHist[i][x], 0, 1, cv2.NORM_MINMAX)
	 
	 
        return (description,npSurfHist)


# For HOG : 
# CELL_DIMENSION is the dimension of the cells on which we will compute local histograms 
# NB_ORIENTATIONS is the number of bins of this local histograms 
# intuitively, if CELL_DIMENSION is small it's better to have a small NB_ORIENTATIONS in order to have meaningful local histograms
# NB_CLUSTERS is the number of bins of the global histograms (the number of clusters in the KMEANS algorithm used for the bag of word)
# MAXITER is the maximum number of iteration for the clustering algorithm

#Takes the npImages and returns a (nbImages, NB_CLUSTERS) np.array with the histograms for each image. 
#Need to specify the number of cores needed for computing, can be set as multiprocessing.cpu_count() in order to use all cpu cores in the system
def calcHOGParallel(nameDB, npImages, CELL_DIMENSION, NB_ORIENTATIONS, NB_CLUSTERS, MAXITER, NB_CORES):
    param = "CellDimension_"+ str(CELL_DIMENSION) +"-"+"nbOrientaions_"+ str(NB_ORIENTATIONS) +"-"+"nbClusters_"+ str(NB_CLUSTERS) +"-"+"Maxiter_"+ str(MAXITER)
    description = nameDB + "-HOG-" + param
    
    HOG = extractHOGFeatureParallel(npImages, CELL_DIMENSION, NB_ORIENTATIONS, NB_CLUSTERS, MAXITER, NB_CORES)
    
    return (description, HOG)

#Takes the npImages and returns a (nbImages, NB_CLUSTERS) np.array with the histograms for each image. 
def calcHOG(nameDB, npImages, CELL_DIMENSION, NB_ORIENTATIONS, NB_CLUSTERS, MAXITER):
    param = "CellDimension_"+ str(CELL_DIMENSION) +"-"+"nbOrientaions_"+ str(NB_ORIENTATIONS) +"-"+"nbClusters_"+ str(NB_CLUSTERS) +"-"+"Maxiter_"+ str(MAXITER)
    description = nameDB + "-HOG-" + param
    
    HOG = extractHOGFeature(npImages, CELL_DIMENSION, NB_ORIENTATIONS, NB_CLUSTERS, MAXITER)
    
    return (description, HOG) 
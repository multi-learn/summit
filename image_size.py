
import os as os        # for iteration throug directories
import pandas as pd # for Series and DataFrames
import cv2          # for OpenCV 
import datetime     # for TimeStamp in CSVFile
from scipy.cluster.vq import * # for Clustering http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
import numpy as np  # for arrays
import time       # for time calculations
from feature_extraction_try import imgCrawl, getClassLabels

def findmaxDim(npImages):
  
  max_height = 0
  max_width = 0
  heights =[]
  widths = []
  totals = []
  count=0
  poulet=0

  for npImage in npImages:
    height, width, channels = cv2.imread(npImage[1]).shape
    heights.append(height)
    widths.append(width)
    totals.append(width*height)
    if width * height > 500000:
    	count+=1
    if not(abs(height-200) <200):
    	poulet+=1
    	print (npImage[1])
    if width > max_width:
      max_width=width

  print float(poulet)*100/len(heights)
  # print float(count)*100/len(heights)
  return heights, widths, totals


path ='../../03-jeux-de-donnees/101_ObjectCategories'

# get dictionary to link classLabels Text to Integers
sClassLabels = getClassLabels(path)

# Get all path from all images inclusive classLabel as Integer
dfImages = imgCrawl(path, sClassLabels)

npImages = dfImages.values
heights = []
heights, widths, totals= findmaxDim(npImages)
heights_ = sorted(list(set(heights)), reverse=True)
widths_ = sorted(list(set(widths)), reverse=True)
totals_ = sorted(totals, reverse=True)
# print (totals_[len(totals_)/2])
# print ("height", sum(heights)/len(heights), "width", sum(widths)/len(widths) )
# print heights_ 
# print("poulmmet")
# print widths_
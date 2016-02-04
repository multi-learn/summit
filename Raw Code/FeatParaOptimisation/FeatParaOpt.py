#!/usr/bin/env python

""" Function to optimise feature parameters """

# Import built-in modules
import time       # for time calculations
import numpy as np# for numpy arrays
import pandas as pd # for Series and DataFrames

# Import 3rd party modules

# Import own modules
import FeatExtraction	# Functions for Feature Extractions#
import ClassifMonoView	# Functions for classification

# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Development" #Production, Development, Prototype
__date__	= 2016-01-23



# dfImages: Database with all images
# feature: which feature? e.g. ColorHistogram
# paramter: which parameter should be tested? e.g. bins of histogram
# valueStart: Value for paramter to start with
# valueEnd: Value for paramter to end with
# nCalculations: How many calculations between valueStart and valueEnd? e.g. vS=0,VE=9,nCalc=10 -> test:0,1,2,3,4,5,6,7,8,9
# boolCV: Boolian if CrossValidation should be used
def perfFeatMonoV(nameDB, dfImages,feature, parameter, valueStart, valueEnd, nCalculations, boolCV):
        # TIME for total calculation
        t_tot_start = time.time()
	
        # Value check - are the given values possible: e.g. bins valueStart = -1 -> error
    
        # Calculate Stepwidth
        if(nCalculations>1):
                step = float(valueEnd-valueStart)/float(nCalculations-1)
                valueArray = np.around(np.array(range(0,nCalculations))*step) + valueStart
	else:
                valueArray = [valueStart]
        # FeatExtraction Results DataFrame
        df_feat_res = pd.DataFrame()
        
        # Array with Calculations for Print
        arr_Calc = range(1,nCalculations+1)
        
        # Initialization
        descriptors = None
        
        # for-loop from valueStart, step stepwidth, until valueEnde
        for valuePara, i in zip(valueArray,arr_Calc):
                print "### Start:\t FeatureExtraction Nr:" + str(i) + " from:" + str(max(arr_Calc)) + " with:" + parameter + "=" + str(valuePara) + " ###"
                # TIME for Extraction
                t_extract_start = time.time()
        
                # Features extraction start
                # Call extraction function with parameters -> returns feature
                if(feature=="RGB"):
                        # Basic Setup
                        numberOfBins = 16
                        maxColorIntensity = 256
                        boolNormMinMax = False
			
                        # ParamaterTest
                        if(parameter=="Bins"):
                                numberOfBins = valuePara
                        elif(parameter=="MaxCI"):
                                maxColorIntensity = valuePara
                        elif(parameter=="Norm"):
                                boolNormMinMax = valuePara
		
                        # Extract Feature from DB
                        feat_desc,f_extr_res = FeatExtraction.calcRGBColorHisto(nameDB, dfImages, numberOfBins, maxColorIntensity, boolNormMinMax)
			
                elif(feature=="HSV"):
                        # Basic Setup
                        h_bins = 8 
                        s_bins = 3
                        v_bins = 3
                        histSize = [h_bins, s_bins, v_bins]
                        boolNormMinMax = False
			
                        # ParamaterTest
                        if(parameter=="Bins"):
                                histSize = valuePara
                        elif(parameter=="Norm"):
                                boolNormMinMax = valuePara
		
                        # Extract Feature from DB
                        feat_desc,f_extr_res = FeatExtraction.calcHSVColorHisto(nameDB, dfImages, histSize, boolNormMinMax)
		
                elif(feature=="SURF"):
                        # Basic Setup
                        cluster = 50
                        boolNormMinMax = False
                        boolSIFT = False
                        
                        # ParamaterTest
                        if(parameter=="Cluster"):
                                cluster = valuePara
                        elif(parameter=="Norm"):
                                boolNormMinMax = valuePara
                                
                        if descriptors is None:
                                descriptors,des_list = FeatExtraction.calcSURFSIFTDescriptors(dfImages, boolSIFT)
		
                        # Extract Feature from DB
                        feat_desc,f_extr_res = FeatExtraction.calcSURFSIFTHisto(nameDB, dfImages, cluster, boolNormMinMax, descriptors, des_list, boolSIFT)
                        
                elif(feature=="SIFT"):
                        # Basic Setup
                        cluster = 50
                        boolNormMinMax = False
                        boolSIFT = True
                        
                        # ParamaterTest
                        if(parameter=="Cluster"):
                                cluster = valuePara
                        elif(parameter=="Norm"):
                                boolNormMinMax = valuePara
                        
                        if descriptors is None:
                                descriptors,des_list = FeatExtraction.calcSURFSIFTDescriptors(dfImages, boolSIFT)
		
                        # Extract Feature from DB
                        feat_desc,f_extr_res = FeatExtraction.calcSURFSIFTHisto(nameDB, dfImages, cluster, boolNormMinMax, descriptors, des_list, boolSIFT)
                else:
                        print "ERROR: Selected Feature does not exist"
                        print "Feature: " + str(feature)
		                
                # TIME for Extraction END
                t_extract = time.time() - t_extract_start
                
                print "### Done:\t FeatureExtraction Nr:" + str(i) + " from:" + str(max(arr_Calc)) + " ###"

                # TIME for CLASSIFICATION
                t_classif_start = time.time()
		
                # Calculate Train/Test data
                #Basic Setup
                split = 0.7
                X_train, X_test, y_train, y_test = ClassifMonoView.calcTrainTest(f_extr_res, dfImages.classLabel, split)
                
                # Own Function for split: ClassifMonoView.calcTrainTestOwn
		       
                # Begin Classification RandomForest
                # call function: return fitted model
                
                print "### Start:\t Classification Nr:" + str(i) + " from:" + str(max(arr_Calc)) + " ###"
        
                # Basic Setup
                num_estimators = [50, 101, 150, 200]
		
                if(boolCV==True):
                        cl_desc, cl_res = ClassifMonoView.calcClassifRandomForestCV(X_train, y_train, num_estimators)
                else:
                        cl_desc, cl_res = ClassifMonoView.calcClassifRandomForest(X_train, X_test, y_test, y_train, num_estimators)
                        
                print "### Done:\t Classification Nr:" + str(i) + " from:" + str(max(arr_Calc)) + " ###"
		
                # TIME for CLASSIFICATION END
                t_classif = time.time() - t_classif_start
        
                # Add result for to Results DF
                df_feat_res = df_feat_res.append({'a_feat_desc': feat_desc, 'b_feat_extr_time':t_extract, 'c_cl_desc': cl_desc, 'd_cl_res': cl_res, 
                                                'e_cl_time': t_classif, 'f_cl_score': cl_res.best_score_}, ignore_index=True)

        # End Loop
	
        # TIME for total calculation END
        t_tot = time.time() - t_tot_start
        
        return df_feat_res
    
    
    


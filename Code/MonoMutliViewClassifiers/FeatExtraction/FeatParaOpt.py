#!/usr/bin/env python

""" Library: Function to optimise feature parameters """

# Import built-in modules
import time                             # for time calculations

# Import 3rd party modules
import numpy as np                      # for numpy arrays
import pandas as pd                     # for Series and DataFrames
import logging                          # To create Log-Files  

# Import own modules
import FeatExtraction	                # Functions for Feature Extractions
import Code.Niko.ClassifMonoView  # Functions for classification

# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Prototype"           # Production, Development, Prototype
__date__	= 2016-03-25


# dfImages:     Database with all images
# feature:      which feature? e.g. ColorHistogram
# para_opt:     optimisation parameters
# para_RGB:     RGB parameters
# para_HSV:     HSV paramters
# para_SIFT:    SIFT parameters
# para_SURF:    SURF parameters
# para_HOG:     HOG paramters
# para_Cl:      Classification parameters
def perfFeatMonoV(nameDB, dfImages, para_opt, para_RGB, para_HSV, para_SIFT, para_SURF, para_HOG, para_Cl):
        # TIME for total calculation
        t_tot_start = time.time()
	
        # Values from Array into variables - easier to read the code
        feature = para_opt[0]
        parameter = para_opt[1]
        valueStart = para_opt[2]
        valueEnd = para_opt[3]
        nCalculations = para_opt[4]
    
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
                valuePara = int(valuePara)
                logging.debug("Start:\t FeatureExtraction Nr:" + str(i) + " from:" + str(max(arr_Calc)) + " with " + parameter + "=" + str(valuePara) + " ###")
                # TIME for Extraction
                t_extract_start = time.time()
                
                
        
                # Features extraction start
                # Call extraction function with parameters -> returns feature
                if(feature=="RGB"):
                        # Basic Setup
                        numberOfBins = para_RGB[0]
                        maxColorIntensity = para_RGB[1]
                        boolNormMinMax = para_RGB[2]
                        
			
                        # ParamaterTest
                        if(parameter=="RGB_Bins"):
                                numberOfBins = valuePara
                        elif(parameter=="RGB_CI"):
                                maxColorIntensity = valuePara
		
                        # Extract Feature from DB
                        feat_desc,f_extr_res = FeatExtraction.calcRGBColorHisto(nameDB, dfImages, numberOfBins, maxColorIntensity, boolNormMinMax)
			
                elif(feature=="HSV"):
                        # Basic Setup
                        h_bins = para_HSV[0]
                        s_bins = para_HSV[1]
                        v_bins = para_HSV[2]
                        
                        boolNormMinMax = para_HSV[3]
			
                        # ParamaterTest
                        if(parameter=="HSV_H_Bins"):
                                h_bins = valuePara
                        elif(parameter=="HSV_S_Bins"):
                                s_bins = valuePara
                        elif(parameter=="HSV_V_Bins"):
                                v_bins = valuePara
                        
                        histSize = [h_bins,s_bins,v_bins]
                        
                        # Extract Feature from DB
                        feat_desc,f_extr_res = FeatExtraction.calcHSVColorHisto(nameDB, dfImages, histSize, boolNormMinMax)
		
                elif(feature=="SIFT"):
                        # Basic Setup
                        cluster = para_SURF[0]
                        boolNormMinMax = para_SURF[1]
                        boolSIFT = True
                        
                        
                        # ParamaterTest
                        if(parameter=="SIFT_Cluster"):
                                cluster = valuePara
                                
                        
                        if descriptors is None:
                                descriptors,des_list = FeatExtraction.calcSURFSIFTDescriptors(dfImages, boolSIFT)
		
                        # Extract Feature from DB
                        feat_desc,f_extr_res = FeatExtraction.calcSURFSIFTHisto(nameDB, dfImages, cluster, boolNormMinMax, descriptors, des_list, boolSIFT)
                
                elif(feature=="SURF"):
                        # Basic Setup
                        cluster = para_SIFT[0]
                        boolNormMinMax = para_SIFT[1]
                        boolSIFT = False
                        
                        
                        # ParamaterTest
                        if(parameter=="SURF_Cluster"):
                                cluster = valuePara
                                
                        if descriptors is None:
                                descriptors,des_list = FeatExtraction.calcSURFSIFTDescriptors(dfImages, boolSIFT)
		
                        # Extract Feature from DB
                        feat_desc,f_extr_res = FeatExtraction.calcSURFSIFTHisto(nameDB, dfImages, cluster, boolNormMinMax, descriptors, des_list, boolSIFT)
                        
                
                elif(feature=="HOG"):                       
                        CELL_DIMENSION = para_HOG[0]
                        NB_ORIENTATIONS = para_HOG[1]
                        NB_CLUSTERS = para_HOG[2]
                        MAXITER = para_HOG[3]
                        NB_CORES = para_HOG[4]
                        
                        # ParamaterTest
                        if(parameter=="HOG_Cluster"):
                                NB_CLUSTERS = valuePara
                        
                        # Extract Feature from DB
                        feat_desc,f_extr_res = FeatExtraction.calcHOGParallel(nameDB, dfImages.values, CELL_DIMENSION, NB_ORIENTATIONS, NB_CLUSTERS, MAXITER, NB_CORES)
                        
                else:
                        print "ERROR: Selected Feature does not exist"
                        print "Feature: " + str(feature)
		                
                # TIME for Extraction END
                t_extract = time.time() - t_extract_start
                
                logging.debug("Done:\t FeatureExtraction Nr:" + str(i) + " from:" + str(max(arr_Calc)) + " ###")

                # TIME for CLASSIFICATION
                t_classif_start = time.time()
		
                # Values from Array into variables - easier to read the code
                split = para_Cl[0]
                num_estimators = para_Cl[1]
                cv_folds = para_Cl[2]
                clas_cores = para_Cl[3]                
                
                # Calculate Train/Test data
                X_train, X_test, y_train, y_test = Code.Niko.ClassifMonoView.calcTrainTest(f_extr_res, dfImages.classLabel, split)
                # Own Function for split: ClassifMonoView.calcTrainTestOwn
		       
                # Begin Classification RandomForest
                # call function: return fitted model
                
                logging.debug("Start:\t Classification Nr:" + str(i) + " from:" + str(max(arr_Calc)))
                
                cl_desc, cl_res = Code.Niko.ClassifMonoView.calcClassifRandomForestCV(X_train, y_train, num_estimators, cv_folds, clas_cores)
                    
                logging.debug("Done:\t Classification Nr:" + str(i) + " from:" + str(max(arr_Calc)))
		
                # TIME for CLASSIFICATION END
                t_classif = time.time() - t_classif_start
        
                # Add result to Results DF
                df_feat_res = df_feat_res.append({'a_feat_desc': feat_desc, 'b_feat_extr_time':t_extract, 'c_cl_desc': cl_desc, 'd_cl_res': cl_res, 
                                                'e_cl_time': t_classif, 'f_cl_score': cl_res.best_score_}, ignore_index=True)

        # End Loop
	
        # TIME for total calculation END
        t_tot = time.time() - t_tot_start
        
        return df_feat_res
    
    
    


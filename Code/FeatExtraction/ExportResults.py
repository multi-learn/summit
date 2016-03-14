#!/usr/bin/env python

""" Functions to generate DB """

# Import built-in modules
import os                               # for iteration throug directories
import string                           # to generate a range of letters

# Import 3rd party modules
import pandas as pd                     # for Series and DataFrames
import numpy as np                      # for Numpy Arrays
import matplotlib.pyplot as plt         # for Plots
from scipy.interpolate import interp1d  # to Interpolate Data
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker # to generate the Annotations in plot

# Import own modules

# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Development" #Production, Development, Prototype
__date__	= 2016-01-23

#### Export Features to CSV
def exportPandasToCSV(pandasSorDF, filename):
        path = os.getcwdu() + "\\" + filename
    
        if os.path.isfile(path + ".csv"):
                for i in range(1,20):
                        testFileName = filename  + "-" + str(i) + ".csv"
                        if os.path.isfile(os.getcwdu() + "\\" +  testFileName)!=True:
                                pandasSorDF.to_csv(testFileName + ".csv", sep=';', decimal=',')
                                break

        else:
                pandasSorDF.to_csv(filename + ".csv", sep=';', decimal=',')


def exportNumpyToCSV(numpyArray, filename, format):
        path = os.getcwdu() + "\\" + filename
    
        if os.path.isfile(path + ".csv"):
                for i in range(1,20):
                        testFileName = filename  + "-" + str(i) + ".csv"
                        if os.path.isfile(os.getcwdu() + "\\" +  testFileName)!=True:
                                np.savetxt(testFileName, numpyArray, delimiter=";", fmt=format)
                                break

        else:
                np.savetxt(filename + ".csv", numpyArray, delimiter=";", fmt=format)
		
		
#### Rendering of results

### Rendering of Score and Time
def showScoreTime(resScore, resTime, rangeX, parameter, feat_desc, cl_desc, fig_desc, y_desc1, y_desc2):
        # Determine interpolated functions
        f_score_interp = interp1d(rangeX, resScore, kind='quadratic')
        f_time_interp  = interp1d(rangeX, resTime, kind='quadratic')
        
        # Figure1 with subplot
        fig, ax1 = plt.subplots()
        
        #plt.plot(x, y, type of line)
        # Generating X-Axis
        xnew = np.linspace(0, max(rangeX), num=100, endpoint=True)
        
        # First Axis for Score (left)
        ax1.plot(rangeX, resScore, 'bo', rangeX, f_score_interp(rangeX), 'b-')
        ax1.set_xlabel(parameter, fontsize=16)
        ax1.set_ylabel(y_desc1, color='b', fontsize=16)
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        
        # First Axis for Time (right)
        ax2 = ax1.twinx()
        ax2.plot(rangeX, resTime, 'ro', rangeX, f_time_interp(rangeX), 'r-')
        ax2.set_ylabel(y_desc2, color='r', fontsize=16)
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        
        
        letters = string.lowercase[0:len(rangeX)]
        legend = ""
        for act_x, act_score, act_time, act_feat_desc, letter, act_cl_desc in zip(rangeX, resScore, resTime, feat_desc,letters,cl_desc):
                # Add a letter (a,b,c,..) to each DataPoint
                ax1.annotate(letter, xy=(act_x, act_score), xytext=(act_x, act_score))
                ax2.annotate(letter, xy=(act_x, act_time), xytext=(act_x, act_time))
                # Creates a legend with description of feature and classificator of each datapoint
                legend = legend + letter + ": " + act_feat_desc + " Classf: " + act_cl_desc + "\n"
                
        # Remove last \n
        legend = legend[:-1]
        
        box1  = TextArea(legend, textprops=dict(color="k"))
        box = HPacker(children=[box1],
                      align="center",
                      pad=0, sep=5)
                      
        anchored_box = AnchoredOffsetbox(loc=3,
                                         child=box, pad=0.2,
                                         frameon=True,
                                         bbox_to_anchor=(0., 1.02),
                                         bbox_transform=ax1.transAxes,
                                         borderpad=1.5,
                                         )
        ax1.add_artist(anchored_box)
        fig.subplots_adjust(top=0.8)

        ax1.legend(['Score Data', 'Score Interpolated'], loc='upper left')
        ax2.legend(['Time Data', 'Time Interpolated'], loc='lower right')
        
        plt.title(fig_desc, fontsize=18)
        
        plt.show()
        


### Result comparision per class
def calcScorePerClass(np_labels, np_output):

        pd_label_test = pd.Series(np_labels)
        pd_output = pd.Series(np_output)
        score = []

        for i in pd_label_test.unique():
                matches = sum(pd_label_test[pd_label_test == i] == pd_output[pd_label_test[pd_label_test == i].index])
                count = float(len(pd_label_test[pd_label_test == i]))
                score.append(matches / count)

        score = np.array(score)
        return score
	
### Bar-Plot for score

def showResults(np_score):
        plt.bar(range(0,len(score)), score*100,1)
        plt.xlabel('ClassLabels')
        plt.ylabel('Score in %')
        plt.title('Resuls of Classification\nCaltech Database')
        plt.axis([0, len(score), 0, 100])
        plt.xticks(range(0,len(score),5))
        plt.show()
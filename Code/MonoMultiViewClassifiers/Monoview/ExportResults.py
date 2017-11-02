#!/usr/bin/env python

""" Library: Functions to export preds to CSV or plots """

# Import built-in modules
import os  # for iteration throug directories
import string  # to generate a range of letters

# Import 3rd party modules
import pandas as pd  # for Series and DataFrames
import numpy as np  # for Numpy Arrays
import matplotlib.pyplot as plt  # for Plots
from scipy.interpolate import interp1d  # to Interpolate Data
import matplotlib

# matplotlib.use('Agg')
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker  # to generate the Annotations in plot
from pylab import rcParams  # to change size of plot
from sklearn import metrics  # For stastics on classification

# Import own modules

# Author-Info
__author__ = "Nikolas Huelsmann"
__status__ = "Prototype"  # Production, Development, Prototype
__date__ = 2016 - 03 - 25


#### Export Features to CSV
def exportPandasToCSV(pandasSorDF, directory, filename):
    file = directory + filename

    # Makes sure that the file does not yet exist
    if os.path.isfile(file + ".csv"):
        for i in range(1, 20):
            testFileName = filename + "-" + str(i) + ".csv"
            if not os.path.isfile(directory + testFileName):
                pandasSorDF.to_csv(directory + testFileName, sep=';')
                break

    else:
        pandasSorDF.to_csv(file + ".csv", sep=';')


def exportNumpyToCSV(numpyArray, directory, filename, format):
    file = directory + filename

    # Makes sure that the file does not yet exist
    if os.path.isfile(file + ".csv"):
        for i in range(1, 20):
            testFileName = filename + "-" + str(i) + ".csv"
            if not os.path.isfile(directory + testFileName):
                np.savetxt(directory + testFileName, numpyArray, delimiter=";", fmt=format)
                break

    else:
        np.savetxt(file + ".csv", numpyArray, delimiter=";", fmt=format)


#### Rendering of results

### Rendering of Score and Time
def showScoreTime(directory, filename, store, resScore, resTime, rangeX, parameter, feat_desc, cl_desc, fig_desc,
                  y_desc1,
                  y_desc2):
    # Determine interpolated functions
    f_score_interp = interp1d(rangeX, resScore, kind='quadratic')
    f_time_interp = interp1d(rangeX, resTime, kind='quadratic')

    # Change size of plot
    rcParams['figure.figsize'] = 20, 10

    # Figure1 with subplot
    fig, ax1 = plt.subplots()

    # plt.plot(x, y, type of line)
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
    for act_x, act_score, act_time, act_feat_desc, letter, act_cl_desc in zip(rangeX, resScore, resTime, feat_desc,
                                                                              letters, cl_desc):
        # Add a letter (a,b,c,..) to each DataPoint
        ax1.annotate(letter, xy=(act_x, act_score), xytext=(act_x, act_score))
        ax2.annotate(letter, xy=(act_x, act_time), xytext=(act_x, act_time))
        # Creates a legend with description of feature and classificator of each datapoint
        legend = legend + letter + ") Feature: " + act_feat_desc + "; Classifier: " + act_cl_desc + "\n"

    # Remove last \n
    legend = legend[:-1]

    box1 = TextArea(legend, textprops=dict(color="k"))
    box = HPacker(children=[box1],
                  align="center",
                  pad=0, sep=5)

    anchored_box = AnchoredOffsetbox(loc=3,
                                     child=box, pad=0.2,
                                     frameon=True,
                                     bbox_to_anchor=(0, 1.04),
                                     # to change the place of the legend (text above of figure)
                                     bbox_transform=ax1.transAxes,
                                     borderpad=1.0,
                                     )
    ax1.add_artist(anchored_box)
    fig.subplots_adjust(top=0.7)

    ax1.legend(['Score Data', 'Score Interpolated'], loc='upper left')
    ax2.legend(['Time Data', 'Time Interpolated'], loc='lower right')

    plt.title(fig_desc, fontsize=18)

    if store:
        # Makes sure that the file does not yet exist
        file = directory + filename

        if os.path.isfile(file + ".png"):
            for i in range(1, 20):
                testFileName = filename + "-" + str(i) + ".png"
                if not os.path.isfile(directory + testFileName):
                    plt.savefig(directory + testFileName)
                    break

        else:
            plt.savefig(file)
    else:
        plt.show()

    plt.close()


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

def showResults(directory, filename, db, feat, score):
    plt.bar(range(0, len(score)), score * 100, 1)
    plt.xlabel('ClassLabels')
    plt.ylabel('Precision in %')
    plt.title('Results of ' + feat + '-Classification\n for ' + db + ' Database')
    plt.axis([0, len(score), 0, 100])
    plt.xticks(range(0, len(score), 5))

    # Makes sure that the file does not yet exist
    file = directory + filename

    if os.path.isfile(file + ".png"):
        for i in range(1, 20):
            testFileName = filename + "-" + str(i) + ".png"
            if not os.path.isfile(directory + testFileName):
                plt.savefig(directory + testFileName)
                break

    else:
        plt.savefig(file)

    plt.close()


    # instead of saving - decomment plt.show()
    # plt.show()


# Function to calculate the accuracy score for test data
def accuracy_score(y_test, y_test_pred):
    return metrics.accuracy_score(y_test, y_test_pred)


# Function to calculate a report of classifiaction and store it
def classification_report_df(directory, filename, y_test, y_test_pred, labels, target_names):
    # Calculate the metrics
    precision, recall, f1score, support = metrics.precision_recall_fscore_support(y_test, y_test_pred, beta=1.0,
                                                                                  labels=labels, pos_label=None,
                                                                                  average=None)

    # turn result into DataFrame
    scores_df = pd.DataFrame(data=[precision, recall, f1score, support])
    scores_df.index = ["Precision", "Recall", "F1", "Support"]
    scores_df.columns = target_names
    scores_df = scores_df.transpose()

    # Store result as CSV
    exportPandasToCSV(scores_df, directory, filename)

    # return the results
    return scores_df


# Function to calculate a report of classifiaction and store it
def confusion_matrix_df(directory, filename, y_test, y_test_pred, target_names):
    # Transform into pd Series
    y_actu = pd.Series(y_test, name='Actual')
    y_pred = pd.Series(y_test_pred, name='Predicted')

    # Calculate confusion matrix
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    # Normalization of confusion matrix
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    df_conf_norm.index = target_names + ['All']
    df_conf_norm.columns = target_names + ['All']

    # Add Row: Actual / Column: Predicted into first cell [0,0]


    # Store result as CSV
    exportPandasToCSV(df_conf_norm, directory, filename)

    return df_conf_norm


def plot_confusion_matrix(directory, filename, df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)  # imshow
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    # plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

    # Makes sure that the file does not yet exist

    file = directory + filename

    if os.path.isfile(file + ".png"):
        for i in range(1, 20):
            testFileName = filename + "-" + str(i) + ".png"
            if not os.path.isfile(directory + testFileName):
                plt.savefig(directory + testFileName)
                break

    else:
        plt.savefig(file)

    plt.close()


def classification_stats(directory, filename, scores_df, acc):
    # Accuracy on test over all classes
    acc = acc

    # Top 10 classes by F1-Score
    top10 = scores_df.sort_values(["F1"], ascending=False).head(10)
    top10 = list(top10.index)

    # Worst 10 classes by F1-Score
    worst10 = scores_df.sort_values(["F1"], ascending=True).head(10)
    worst10 = list(worst10.index)

    # Ratio of classes with F1-Score==0 of all classes
    ratio_zero = float(float(len(scores_df[scores_df.F1 == 0])) / float(len(scores_df)))

    # Mean of F1-Score of top 10 classes by F1-Score
    mean_10 = np.mean(scores_df.sort_values(["F1"], ascending=False).head(10).F1)

    # Mean of F1-Score of top 20 classes by F1-Score
    mean_20 = np.mean(scores_df.sort_values(["F1"], ascending=False).head(20).F1)

    # Mean of F1-Score of top 30 classes by F1-Score
    mean_30 = np.mean(scores_df.sort_values(["F1"], ascending=False).head(30).F1)

    # Create DataFrame with stats
    d = {'Statistic': ['Accuracy score on test', 'Top 10 classes by F1-Score', 'Worst 10 classes by F1-Score',
                       'Ratio of classes with F1-Score==0 of all classes',
                       'Mean of F1-Score of top 10 classes by F1-Score',
                       'Mean of F1-Score of top 20 classes by F1-Score',
                       'Mean of F1-Score of top 30 classes by F1-Score'],
         'Values': [acc, top10, worst10, ratio_zero, mean_10, mean_20, mean_30]}
    df_stats = pd.DataFrame(d)

    # Store result as CSV
    exportPandasToCSV(df_stats, directory, filename)

    # return pandas
    return df_stats

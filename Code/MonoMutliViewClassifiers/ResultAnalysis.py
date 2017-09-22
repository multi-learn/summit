# Import built-in modules
import time
import pylab
import logging

# Import third party modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib as mpl

#Import own Modules
import Metrics
from utils.Transformations import signLabels

# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype

def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                "%.2f" % height,
                ha='center', va='bottom')


def genNamesFromRes(mono, multi):
    names = [res[1][0]+"-"+res[1][1][-1] for res in mono]
    names+=[type_ for type_, a, b, c in multi if type_ != "Fusion"]
    names+=[ "Late-"+str(a["fusionMethod"]) for type_, a, b, c in multi if type_ == "Fusion" and a["fusionType"] != "EarlyFusion"]
    names+=[ "Early-"+a["fusionMethod"]+"-"+a["classifiersNames"][0]  for type_, a, b, c in multi if type_ == "Fusion" and a["fusionType"] != "LateFusion"]
    return names


def resultAnalysis(benchmark, results, name, times, metrics):
    mono, multi = results
    for metric in metrics:
        names = genNamesFromRes(mono, multi)
        nbResults = len(mono)+len(multi)
        validationScores = [float(res[1][2][metric[0]][0]) for res in mono]
        validationScores += [float(scores[metric[0]][0]) for a, b, scores, c in multi]
        validationSTD = [float(res[1][2][metric[0]][2]) for res in mono]
        validationSTD += [float(scores[metric[0]][2]) for a, b, scores, c in multi]
        trainScores = [float(res[1][2][metric[0]][1]) for res in mono]
        trainScores += [float(scores[metric[0]][1]) for a, b, scores, c in multi]
        trainSTD = [float(res[1][2][metric[0]][3]) for res in mono]
        trainSTD += [float(scores[metric[0]][3]) for a, b, scores, c in multi]
        f = pylab.figure(figsize=(40, 30))
        width = 0.35       # the width of the bars
        fig = plt.gcf()
        fig.subplots_adjust(bottom=105.0, top=105.01)
        ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
        if metric[1]!=None:
            metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
        else:
            metricKWARGS = {}
        ax.set_title(getattr(Metrics, metric[0]).getConfig(**metricKWARGS)+" on validation set for each classifier")
        rects = ax.bar(range(nbResults), validationScores, width, color="r", yerr=validationSTD)
        rect2 = ax.bar(np.arange(nbResults)+width, trainScores, width, color="0.7", yerr=trainSTD)
        autolabel(rects, ax)
        autolabel(rect2, ax)
        ax.legend((rects[0], rect2[0]), ('Train', 'Test'))
        ax.set_xticks(np.arange(nbResults)+width)
        ax.set_xticklabels(names, rotation="vertical")

        f.savefig("Results/"+time.strftime("%Y%m%d-%H%M%S")+"-"+name+"-"+metric[0]+".png")
    logging.info("Extraction time : "+str(times[0])+"s, Monoview time : "+str(times[1])+"s, Multiview Time : "+str(times[2])+"s")


def analyzeLabels(labelsArrays, realLabels, results):
    mono, multi = results
    classifiersNames = genNamesFromRes(mono, multi)
    nbClassifiers = len(classifiersNames)
    nbExamples = realLabels.shape[0]
    nbIter = 20
    data = np.zeros((nbExamples, nbClassifiers*nbIter))
    tempData = np.array([labelsArray == realLabels for labelsArray in np.transpose(labelsArrays)]).astype(int)
    for classifierIndex in range(nbClassifiers):
        for iterIndex in range(nbIter):
            data[:,classifierIndex*nbIter+iterIndex] = tempData[classifierIndex,:]
    fig = pylab.figure(figsize=(30,20))
    cmap = mpl.colors.ListedColormap(['red','green'])
    bounds=[-0.5,0.5,1.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    cax = plt.imshow(data, interpolation='nearest', cmap=cmap, norm=norm)
    plt.title('Error on examples depending on the classifier')
    ticks = np.arange(0, nbClassifiers*nbIter, nbIter)
    labels = classifiersNames
    plt.xticks(ticks, labels, rotation="vertical")
    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Wrong', ' Right'])
    fig.savefig("Results/"+time.strftime("%Y%m%d-%H%M%S")+"error_analysis.png")
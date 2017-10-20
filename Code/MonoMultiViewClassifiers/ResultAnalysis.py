# Import built-in modules
import time
import pylab
import logging

# Import third party modules
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import cm
import matplotlib as mpl

# Import own Modules
# import Metrics
# from utils.Transformations import signLabels

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                "%.2f" % height,
                ha='center', va='bottom')


def genFusionName(type_, a, b, c):
    if type_ == "Fusion" and a["fusionType"] != "EarlyFusion":
        return "Late-" + str(a["fusionMethod"])
    elif type_ == "Fusion" and a["fusionType"] != "LateFusion":
        return "Early-" + a["fusionMethod"] + "-" + a["classifiersNames"]


def genNamesFromRes(mono, multi):
    names = [res[1][0] + "-" + res[1][1][-1] for res in mono]
    names += [type_ if type_ != "Fusion" else genFusionName(type_, a, b, c) for type_, a, b, c in multi]
    return names


def resultAnalysis(benchmark, results, name, times, metrics, directory, minSize=10):
    mono, multi = results
    for metric in metrics:
        names = genNamesFromRes(mono, multi)
        nbResults = len(mono) + len(multi)
        validationScores = [float(res[1][2][metric[0]][1]) for res in mono]
        validationScores += [float(scores[metric[0]][1]) for a, b, scores, c in multi]
        trainScores = [float(res[1][2][metric[0]][0]) for res in mono]
        trainScores += [float(scores[metric[0]][0]) for a, b, scores, c in multi]

        validationScores = np.array(validationScores)
        trainScores = np.array(trainScores)
        names = np.array(names)
        size = nbResults
        if nbResults < minSize:
            size = minSize
        figKW = {"figsize" : (size, 3.0/4*size+2.0)}
        f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
        barWidth= 0.35
        sorted_indices = np.argsort(validationScores)
        validationScores = validationScores[sorted_indices]
        trainScores = trainScores[sorted_indices]
        names = names[sorted_indices]

        ax.set_title(metric[0] + "\n on validation set for each classifier")
        rects = ax.bar(range(nbResults), validationScores, barWidth, color="r", )
        rect2 = ax.bar(np.arange(nbResults) + barWidth, trainScores, barWidth, color="0.7", )
        autolabel(rects, ax)
        autolabel(rect2, ax)
        ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks(np.arange(nbResults) + barWidth)
        ax.set_xticklabels(names, rotation="vertical")
        plt.tight_layout()
        f.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-" + name + "-" + metric[0] + ".png")
        plt.close()


def analyzeIterLabels(labelsAnalysisList, directory, classifiersNames, minSize=10):
    nbExamples = labelsAnalysisList[0].shape[0]
    nbClassifiers = len(classifiersNames)
    nbIter = 2

    figWidth = max(nbClassifiers / 2, minSize)
    figHeight = max(nbExamples / 20, minSize)
    figKW = {"figsize": (figWidth, figHeight)}
    fig, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    data = sum(labelsAnalysisList)
    cax = plt.imshow(data, interpolation='none', cmap="grey", aspect='auto')
    plt.title('Errors depending on the classifier')
    ticks = np.arange(nbIter/2-0.5, nbClassifiers * nbIter, nbIter)
    plt.xticks(ticks, classifiersNames, rotation="vertical")
    cbar = fig.colorbar(cax, ticks=[0, len(labelsAnalysisList)])
    cbar.ax.set_yticklabels(['Always Wrong', 'Always Right'])
    fig.tight_layout()
    fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-error_analysis.png")
    plt.close()


def analyzeLabels(labelsArrays, realLabels, results, directory, minSize = 10):
    mono, multi = results
    classifiersNames = genNamesFromRes(mono, multi)
    nbClassifiers = len(classifiersNames)
    nbExamples = realLabels.shape[0]
    nbIter = 2
    data = np.zeros((nbExamples, nbClassifiers * nbIter))
    tempData = np.array([labelsArray == realLabels for labelsArray in np.transpose(labelsArrays)]).astype(int)
    for classifierIndex in range(nbClassifiers):
        for iterIndex in range(nbIter):
            data[:, classifierIndex * nbIter + iterIndex] = tempData[classifierIndex, :]
    figWidth = max(nbClassifiers/2, minSize)
    figHeight = max(nbExamples/20, minSize)
    figKW = {"figsize":(figWidth, figHeight)}
    fig, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    cmap = mpl.colors.ListedColormap(['red', 'green'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    cax = plt.imshow(data, interpolation='none', cmap=cmap, norm=norm, aspect='auto')
    plt.title('Errors depending on the classifier')
    ticks = np.arange(nbIter/2-0.5, nbClassifiers * nbIter, nbIter)
    labels = classifiersNames
    plt.xticks(ticks, labels, rotation="vertical")
    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Wrong', ' Right'])
    fig.tight_layout()
    fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-error_analysis.png")
    plt.close()
    return data


def genScoresNames(iterResults, metric, nbResults, names, nbMono, minSize=10):
    nbIter = len(iterResults)
    validationScores = np.zeros((nbIter, nbResults))
    trainScores = np.zeros((nbIter, nbResults))
    for iterIndex, iterResult in enumerate(iterResults):
        mono, multi = iterResult
        validationScores[iterIndex, :nbMono] = np.array([float(res[1][2][metric[0]][1]) for res in mono])
        validationScores[iterIndex, nbMono:] = np.array([float(scores[metric[0]][1]) for a, b, scores, c in multi])
        trainScores[iterIndex, :nbMono] = np.array([float(res[1][2][metric[0]][0]) for res in mono])
        trainScores[iterIndex, nbMono:] = np.array([float(scores[metric[0]][0]) for a, b, scores, c in multi])

    validationSTDs = np.std(validationScores, axis=0)
    trainSTDs = np.std(trainScores, axis=0)
    validationMeans = np.mean(validationScores, axis=0)
    trainMeans = np.mean(trainScores, axis=0)
    size=nbResults
    if nbResults<minSize:
        size=minSize
    figKW = {"figsize" : (size, 3.0/4*size+2.0)}
    f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    barWidth = 0.35  # the width of the bars
    sorted_indices = np.argsort(validationMeans)
    validationMeans = validationMeans[sorted_indices]
    validationSTDs = validationSTDs[sorted_indices]
    trainSTDs = trainSTDs[sorted_indices]
    trainMeans = trainMeans[sorted_indices]
    names = np.array(names)[sorted_indices]

    ax.set_title(metric[0] + " for each classifier")
    rects = ax.bar(range(nbResults), validationMeans, barWidth, color="r", yerr=validationSTDs)
    rect2 = ax.bar(np.arange(nbResults) + barWidth, trainMeans, barWidth, color="0.7", yerr=trainSTDs)
    autolabel(rects, ax)
    autolabel(rect2, ax)
    ax.set_ylim(-0.1, 1.1)
    ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
    ax.set_xticks(np.arange(nbResults) + barWidth)
    ax.set_xticklabels(names, rotation="vertical")
    f.tight_layout()

    return f


def analyzeIterResults(iterResults, name, metrics, directory):
    nbResults = len(iterResults[0][0]) + len(iterResults[0][1])
    nbMono = len(iterResults[0][0])
    nbIter = len(iterResults)
    names = genNamesFromRes(iterResults[0][0], iterResults[0][1])
    for metric in metrics:
        figure = genScoresNames(iterResults, metric, nbResults, names, nbMono)
        figure.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-" + name + "-Mean_on_"
                       + str(nbIter) + "_iter-" + metric[0] + ".png")

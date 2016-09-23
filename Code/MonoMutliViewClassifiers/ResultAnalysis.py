# Import built-in modules
import time
import pylab
import logging

# Import third party modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Import own Modules
import Metrics

# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype

def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                "%.2f" % round(height, 4),
                ha='center', va='bottom')


def resultAnalysis(benchmark, results, name, times, metrics):
    for metric in metrics:
        mono, multi = results
        names = [res[1][0]+"-"+res[1][1][-1] for res in mono]
        names+=[type_ for type_, a, b in multi if type_ != "Fusion"]
        names+=[ "Late-"+str(a["fusionMethod"]) for type_, a, b in multi if type_ == "Fusion" and a["fusionType"] != "EarlyFusion"]
        names+=[ "Early-"+a["fusionMethod"]+"-"+a["classifiersNames"][0]  for type_, a, b in multi if type_ == "Fusion" and a["fusionType"] != "LateFusion"]
        nbResults = len(mono)+len(multi)
        validationScores = [float(res[1][2][metric[0]][2]) for res in mono]
        validationScores += [float(scores[metric[0]][2]) for a, b, scores in multi]
        f = pylab.figure(figsize=(40, 30))
        fig = plt.gcf()
        fig.subplots_adjust(bottom=105.0, top=105.01)
        ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
        if metric[1]!=None:
            metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
        else:
            metricKWARGS = {}
        ax.set_title(getattr(Metrics, metric[0]).getConfig(**metricKWARGS)+" on validation set for each classifier")
        rects = ax.bar(range(nbResults), validationScores, align='center')
        autolabel(rects, ax)
        ax.set_xticks(range(nbResults))
        ax.set_xticklabels(names, rotation="vertical")

        f.savefig("Results/"+time.strftime("%Y%m%d-%H%M%S")+"-"+name+"-"+metric[0]+".png")
    logging.info("Extraction time : "+str(times[0])+"s, Monoview time : "+str(times[1])+"s, Multiview Time : "+str(times[2])+"s")



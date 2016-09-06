# Import built-in modules
import time
import pylab
import logging

# Import third party modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


def resultAnalysis(benchmark, results, name, times, metrics):
    for metric in metrics:
        mono, multi = results
        names = [res[1][0]+"-"+res[1][1][-1] for res in mono]
        names+=[type_ if type_ != "Fusion" else a["fusionType"]+"-"+a["fusionMethod"] for type_, a, b in multi]
        nbResults = len(mono)+len(multi)
        validationScores = [float(res[1][2][metric[0]][2]) for res in mono]
        validationScores += [float(scores[metric[0]][2]) for a, b, scores in multi]
        f = pylab.figure(figsize=(40, 30))
        fig = plt.gcf()
        fig.subplots_adjust(bottom=105.0, top=105.01)
        ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_title(metric[0]+" on validation set for each classifier")
        ax.bar(range(nbResults), validationScores, align='center')
        ax.set_xticks(range(nbResults))
        ax.set_xticklabels(names, rotation="vertical")

        f.savefig("Results/"+name+"-"+metric[0]+"-"+time.strftime("%Y%m%d-%H%M%S")+".png")
    logging.info("Extraction time : "+str(times[0])+"s, Monoview time : "+str(times[1])+"s, Multiview Time : "+str(times[2])+"s")



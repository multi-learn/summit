# Import built-in modules
import time
import pylab

# Import third party modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


def resultAnalysis(benchmark, results, name):
    mono, multi = results
    names = [res[1][0]+res[1][3] for res in mono]
    names+=[type_ if type_ != "Fusion" else a["fusionType"]+a["fusionMethod"] for type_, a, b, c, d in multi]
    nbResults = len(mono)+len(multi)
    accuracies = [100*float(res[1][1]) for res in mono]
    accuracies += [float(accuracy) for a, b, c, d, accuracy in multi]
    f = pylab.figure(figsize=(40, 30))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=105.0, top=105.01)
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title("Accuracies on validation set for each classifier")
    ax.bar(range(nbResults), accuracies, align='center')
    ax.set_xticks(range(nbResults))
    ax.set_xticklabels(names, rotation="vertical")

    f.savefig("Results/"+name+time.strftime("%Y%m%d-%H%M%S")+".png")



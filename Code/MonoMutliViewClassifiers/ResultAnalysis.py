import matplotlib.pyplot as plt
import time
import pylab

def resultAnalysis(benchmark, results):
    mono, multi = results
    names = [type_+feat for [type_, b, c, feat] in mono]+[type_ if type_ != "Fusion" else type_+a["FusionType"]+a["FusionMethod"] for type_, a, b, c, d in multi]
    nbResults = len(mono)+len(multi)
    accuracies = [float(accuracy)*100 for [a, accuracy, c, d] in mono]+[float(accuracy)*100 for a, b, c, d, accuracy in multi]
    f = pylab.figure()
    try:
        fig = plt.gcf()
        fig.subplots_adjust(bottom=2.0)
    except:
        pass
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title("Accuracies on validation set for each classifier")
    ax.bar(range(nbResults), accuracies, align='center')
    ax.set_xticks(range(nbResults))
    ax.set_xticklabels(names, rotation="vertical")

    f.savefig("Results/poulet"+time.strftime("%Y%m%d-%H%M%S")+".png")



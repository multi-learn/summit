import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pylab

def resultAnalysis(benchmark, results):
    mono, multi = results
    flattenedMono = []
    for view in mono:
        for res in view:
            flattenedMono.append(res)
    names = [res[0]+res[3] for res in flattenedMono]
    names+=[type_ if type_ != "Fusion" else type_+a["fusionType"]+a["fusionMethod"] for type_, a, b, c, d in multi]
    nbResults = len(flattenedMono)+len(multi)
    accuracies = [float(res[1]) for res in flattenedMono]
    accuracies += [float(accuracy) for a, b, c, d, accuracy in multi]
    f = pylab.figure(figsize=(40, 30))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=105.0, top=105.01)
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title("Accuracies on validation set for each classifier")
    ax.bar(range(nbResults), accuracies, align='center')
    ax.set_xticks(range(nbResults))
    ax.set_xticklabels(names, rotation="vertical")

    f.savefig("Results/poulet"+time.strftime("%Y%m%d-%H%M%S")+".png")



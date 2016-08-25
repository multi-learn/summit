import matplotlib.pyplot as plt
import time
import pylab

def resultAnalysis(benchmark, results):
    mono, multi = results
    names = [type_+feat for [type_, b, c, feat] in mono]+[type_ if type_ != "Fusion" else type_+a["FusionType"]+a["FusionMethod"] for type_, a, b, c, d in multi]
    nbResults = len(mono)+len(multi)
    accuracies = [float(accuracy)*100 for [a, accuracy, c, d] in mono]+[float(accuracy)*100 for a, b, c, d, accuracy in multi]
    f = pylab.figure()
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title("Accuracies on validation set for each classifier")
    ax.bar(range(nbResults), accuracies, align='center')
    ax.set_xticks(range(nbResults))
    ax.set_xticklabels(names, rotation="vertical")
    try:
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.8)
    except:
        pass
    # plt.bar(range(nbResults), accuracies, 1)
    # plt.xlabel('ClassLabels')
    # plt.ylabel('Precision in %')
    # plt.title('Results of benchmark-Classification')
    # plt.axis([0, nbResults, 0, 100])
    # plt.xticks(range(nbResults), rotation="vertical")

    # Makes sure that the file does not yet exist
    f.savefig("Results/poulet"+time.strftime("%Y%m%d-%H%M%S")+".png")

    #plt.close()



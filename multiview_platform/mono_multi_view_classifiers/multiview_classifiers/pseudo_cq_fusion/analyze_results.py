from ...multiview import analyze_results

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def execute(classifier, trainLabels,
            testLabels, DATASET,
            classificationKWARGS, classificationIndices,
            labels_dictionary, views, nbCores, times,
            name, KFolds,
            hyper_param_search, nIter, metrics,
            views_indices, randomState, labels, classifierModule):
    return analyze_results.execute(classifier, trainLabels,
                                   testLabels, DATASET,
                                   classificationKWARGS, classificationIndices,
                                   labels_dictionary, views, nbCores, times,
                                   name, KFolds,
                                   hyper_param_search, nIter, metrics,
                                   views_indices, randomState, labels, classifierModule)
from ...multiview import analyze_results

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def execute(classifier, trainLabels,
            testLabels, DATASET,
            classificationKWARGS, classificationIndices,
            LABELS_DICTIONARY, views, nbCores, times,
            name, KFolds,
            hyperParamSearch, nIter, metrics,
            viewsIndices, randomState, labels, classifierModule):
    return analyze_results.execute(classifier, trainLabels,
                                   testLabels, DATASET,
                                   classificationKWARGS, classificationIndices,
                                   LABELS_DICTIONARY, views, nbCores, times,
                                   name, KFolds,
                                   hyperParamSearch, nIter, metrics,
                                   viewsIndices, randomState, labels, classifierModule)
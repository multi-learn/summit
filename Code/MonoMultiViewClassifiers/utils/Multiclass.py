import numpy as np
import itertools


def genMulticlassLabels(labels, multiclassMethod, classificationIndices):
    if multiclassMethod == "oneVersusOne":
        nbLabels = len(set(list(labels)))
        if nbLabels == 2:
            return [labels], [(0,1)], [classificationIndices]
        else:
            combinations = itertools.combinations(np.arange(nbLabels), 2)
            multiclassLabels = []
            labelsIndices = []
            oldIndicesMulticlass = []
            for combination in combinations:
                labelsIndices.append(combination)
                oldIndices = [exampleIndex
                              for exampleIndex, exampleLabel in enumerate(labels)
                              if exampleLabel in combination]
                oldTrainIndices = [[oldIndex for oldIndex in oldIndicesMulticlass if oldIndex in trainIndices]
                                   for trainIndices, testIndices in classificationIndices]
                oldTestIndices = [[oldIndex for oldIndex in oldIndicesMulticlass if oldIndex in testIndices]
                                  for trainIndices, testIndices in classificationIndices]
                oldIndicesMulticlass.append([oldTrainIndices, oldTestIndices])
                multiclassLabels.append(np.array([1 if exampleLabel == combination[0]
                                                  else 0
                                                  for exampleLabel in labels[oldIndices]]))
    elif multiclassMethod == "oneVersusRest":
        # TODO : Implement one versus rest if probas are not a problem anymore
        pass
    return multiclassLabels, labelsIndices, oldIndicesMulticlass
#
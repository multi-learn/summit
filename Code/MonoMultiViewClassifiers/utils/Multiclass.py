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
            indicesMulticlass = []
            for combination in combinations:
                labelsIndices.append(combination)
                oldIndices = [exampleIndex
                              for exampleIndex, exampleLabel in enumerate(labels)
                              if exampleLabel in combination]
                trainIndices = [np.array([oldIndex for oldIndex in oldIndices if oldIndex in iterIndices[0]])
                                   for iterIndices in classificationIndices]
                testIndices = [np.array([oldIndex for oldIndex in oldIndices if oldIndex in iterindices[1]])
                                  for iterindices in classificationIndices]
                indicesMulticlass.append([trainIndices, testIndices])
                newLabels = np.zeros(len(labels), dtype=int)-100
                for labelIndex, label in enumerate(labels):
                    if label == combination[0]:
                        newLabels[labelIndex] = 1
                    elif label == combination[1]:
                        newLabels[labelIndex] = 0
                    else:
                        pass
                multiclassLabels.append(newLabels)
    elif multiclassMethod == "oneVersusRest":
        # TODO : Implement one versus rest if probas are not a problem anymore
        pass
    return multiclassLabels, labelsIndices, indicesMulticlass
#
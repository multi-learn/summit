import itertools

import numpy as np


def genMulticlassLabels(labels, multiclassMethod, splits):
    r"""Used to gen the train/test splits and to set up the framework of the adaptation of a multiclass dataset
    to biclass algorithms.

    First, the function checks whether the dataset is really multiclass.

    Then, it generates all the possible couples of different labels in order to perform one versus one classification.

    For each combination, it selects the examples in the training sets (for each statistical iteration) that have their
    label in the combination and does the same for the testing set. It also saves the multiclass testing set in order to
    use multiclass metrics on the decisions.

    Lastly, it creates a new array of biclass labels (0/1) for the biclass classifications used in oneVersusOne

    Parameters
    ----------
    labels : numpy.ndarray
        Name of the database.
    multiclassMethod : string
        The name of the multiclass method used (oneVersusOne, oneVersusAll, ...).
    splits : list of lists of numpy.ndarray
        For each statistical iteration a couple of numpy.ndarrays is stored with the indices for the training set and
        the ones of the testing set.

    Returns
    -------
    multiclassLabels : list of lists of numpy.ndarray
        For each label couple, for each statistical iteration a triplet of numpy.ndarrays is stored with the
        indices for the biclass training set, the ones for the biclass testing set and the ones for the
        multiclass testing set.

    labelsIndices : list of lists of numpy.ndarray
        Each original couple of different labels.

    indicesMulticlass : list of lists of numpy.ndarray
        For each combination, contains a biclass labels numpy.ndarray with the 0/1 labels of combination.
    """
    if multiclassMethod == "oneVersusOne":
        nbLabels = len(set(list(labels)))
        if nbLabels == 2:
            splits = [[trainIndices for trainIndices, _ in splits],
                      [testIndices for _, testIndices in splits],
                      [[] for _ in splits]]
            return [labels], [(0, 1)], [splits]
        else:
            combinations = itertools.combinations(np.arange(nbLabels), 2)
            multiclassLabels = []
            labelsIndices = []
            indicesMulticlass = []
            for combination in combinations:
                labelsIndices.append(combination)
                oldIndices = [exampleIndex
                              for exampleIndex, exampleLabel in
                              enumerate(labels)
                              if exampleLabel in combination]
                trainIndices = [np.array([oldIndex for oldIndex in oldIndices if
                                          oldIndex in iterIndices[0]])
                                for iterIndices in splits]
                testIndices = [np.array([oldIndex for oldIndex in oldIndices if
                                         oldIndex in iterindices[1]])
                               for iterindices in splits]
                testIndicesMulticlass = [np.array(iterindices[1]) for
                                         iterindices in splits]
                indicesMulticlass.append(
                    [trainIndices, testIndices, testIndicesMulticlass])
                newLabels = np.zeros(len(labels), dtype=int) - 100
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


def genMulticlassMonoviewDecision(monoviewResult, classificationIndices):
    learningIndices, validationIndices, testIndicesMulticlass = classificationIndices
    multiclassMonoviewDecisions = monoviewResult.full_labels_pred
    multiclassMonoviewDecisions[
        testIndicesMulticlass] = monoviewResult.y_test_multiclass_pred
    return multiclassMonoviewDecisions


def isBiclass(multiclass_preds):
    if multiclass_preds[0] is []:
        return True
    else:
        return False

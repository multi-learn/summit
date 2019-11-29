from datetime import timedelta as hms

from .. import metrics


def getDBConfigString(name, feat, classification_indices, shape,
                      classLabelsNames, KFolds):
    """
    
    Parameters
    ----------
    name
    feat
    classification_indices
    shape
    classLabelsNames
    KFolds

    Returns
    -------

    """
    learningRate = float(len(classification_indices[0])) / (
                len(classification_indices[0]) + len(classification_indices[1]))
    dbConfigString = "Database configuration : \n"
    dbConfigString += "\t- Database name : " + name + "\n"
    dbConfigString += "\t- View name : " + feat + "\t View shape : " + str(
        shape) + "\n"
    dbConfigString += "\t- Learning Rate : " + str(learningRate) + "\n"
    dbConfigString += "\t- Labels used : " + ", ".join(classLabelsNames) + "\n"
    dbConfigString += "\t- Number of cross validation folds : " + str(
        KFolds.n_splits) + "\n\n"
    return dbConfigString


def getClassifierConfigString(gridSearch, nbCores, nIter, clKWARGS, classifier,
                              output_file_name, y_test):
    classifierConfigString = "Classifier configuration : \n"
    classifierConfigString += "\t- " + classifier.getConfig()[5:] + "\n"
    classifierConfigString += "\t- Executed on " + str(nbCores) + " core(s) \n"
    if gridSearch:
        classifierConfigString += "\t- Got configuration using randomized search with " + str(
            nIter) + " iterations \n"
    classifierConfigString += "\n\n"
    classifierInterpretString = classifier.getInterpret(output_file_name, y_test)
    return classifierConfigString, classifierInterpretString


def getMetricScore(metric, y_train, y_train_pred, y_test, y_test_pred):
    metricModule = getattr(metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in
                            enumerate(metric[1]))
    else:
        metricKWARGS = {}
    metricScoreTrain = metricModule.score(y_train, y_train_pred)
    metricScoreTest = metricModule.score(y_test, y_test_pred)
    metricScoreString = "\tFor " + metricModule.getConfig(
        **metricKWARGS) + " : "
    metricScoreString += "\n\t\t- Score on train : " + str(metricScoreTrain)
    metricScoreString += "\n\t\t- Score on test : " + str(metricScoreTest)
    metricScoreString += "\n"
    return metricScoreString, [metricScoreTrain, metricScoreTest]


def execute(name, learningRate, KFolds, nbCores, gridSearch, metrics_list, nIter,
            feat, CL_type, clKWARGS, classLabelsNames,
            shape, y_train, y_train_pred, y_test, y_test_pred, time,
            random_state, classifier, output_file_name):
    metricsScores = {}
    metricModule = getattr(metrics, metrics_list[0][0])
    trainScore = metricModule.score(y_train, y_train_pred)
    testScore = metricModule.score(y_test, y_test_pred)
    stringAnalysis = "Classification on " + name + " database for " + feat + " with " + CL_type + ".\n\n"
    stringAnalysis += metrics_list[0][0] + " on train : " + str(trainScore) + "\n" + \
                      metrics_list[0][0] + " on test : " + str(
        testScore) + "\n\n"
    stringAnalysis += getDBConfigString(name, feat, learningRate, shape,
                                        classLabelsNames, KFolds)
    classifierConfigString, classifierIntepretString = getClassifierConfigString(
        gridSearch, nbCores, nIter, clKWARGS, classifier, output_file_name, y_test)
    stringAnalysis += classifierConfigString
    for metric in metrics_list:
        metricString, metricScore = getMetricScore(metric, y_train,
                                                   y_train_pred, y_test,
                                                   y_test_pred)
        stringAnalysis += metricString
        metricsScores[metric[0]] = metricScore
        # stringAnalysis += getMetricScore(metric, y_train, y_train_pred, y_test, y_test_pred)
        # if metric[1] is not None:
        #     metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
        # else:
        #     metricKWARGS = {}
        # metricsScores[metric[0]] = [getattr(metrics, metric[0]).score(y_train, y_train_pred),
        #                             getattr(metrics, metric[0]).score(y_test, y_test_pred)]
    stringAnalysis += "\n\n Classification took " + str(hms(seconds=int(time)))
    stringAnalysis += "\n\n Classifier Interpretation : \n"
    stringAnalysis += classifierIntepretString

    imageAnalysis = {}
    return stringAnalysis, imageAnalysis, metricsScores

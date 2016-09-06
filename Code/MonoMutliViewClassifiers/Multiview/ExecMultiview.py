import sys
import os.path

sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from Multiview import *

import GetMultiviewDb as DB
import argparse
import numpy as np
import datetime
import os
import logging
import time
import h5py


# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype



def ExecMultiview_multicore(coreIndex, name, learningRate, nbFolds, databaseType, path, LABELS_DICTIONARY ,
                            gridSearch=False, nbCores=1, metrics=None, nIter=30, **arguments):
    DATASET = h5py.File(path+name+str(coreIndex)+".hdf5", "r")
    return ExecMultiview(DATASET, name, learningRate, nbFolds, 1, databaseType, path, LABELS_DICTIONARY,
                         gridSearch=gridSearch, metrics=None, nIter=30, **arguments)


def ExecMultiview(DATASET, name, learningRate, nbFolds, nbCores, databaseType, path, LABELS_DICTIONARY,
                  gridSearch=False, metrics=None, nIter=30, **kwargs):

    datasetLength = DATASET.get("Metadata").attrs["datasetLength"]
    NB_VIEW = DATASET.get("Metadata").attrs["nbView"]
    views = [str(DATASET.get("View"+str(viewIndex)).attrs["name"]) for viewIndex in range(NB_VIEW)]
    NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]
    if not metrics:
        metrics = [["accuracy_score", None]]

    CL_type = kwargs["CL_type"]
    views = kwargs["views"]
    NB_VIEW = kwargs["NB_VIEW"]
    LABELS_NAMES = kwargs["LABELS_NAMES"]
    classificationKWARGS = kwargs[CL_type+"KWARGS"]

    t_start = time.time()
    logging.info("### Main Programm for Multiview Classification")
    logging.info("### Classification - Database : " + str(name) + " ; Views : " + ", ".join(views) +
                 " ; Algorithm : " + CL_type + " ; Cores : " + str(nbCores))

    for viewIndex in range(NB_VIEW):
        logging.info("Info:\t Shape of " + str(DATASET.get("View"+str(viewIndex)).attrs["name"]) + " :" + str(
            DATASET.get("View"+str(viewIndex)).shape))
    logging.info("Done:\t Read Database Files")


    logging.info("Start:\t Determine validation split for ratio " + str(learningRate))
    validationIndices = DB.splitDataset(DATASET, learningRate, datasetLength)
    learningIndices = [index for index in range(datasetLength) if index not in validationIndices]
    classificationSetLength = len(learningIndices)
    logging.info("Done:\t Determine validation split")

    logging.info("Start:\t Determine "+str(nbFolds)+" folds")
    if nbFolds != 1:
        kFolds = DB.getKFoldIndices(nbFolds, DATASET.get("labels")[...], NB_CLASS, learningIndices)
    else:
        kFolds = [[], range(classificationSetLength)]
    logging.info("Info:\t Length of Learning Sets: " + str(classificationSetLength - len(kFolds[0])))
    logging.info("Info:\t Length of Testing Sets: " + str(len(kFolds[0])))
    logging.info("Info:\t Length of Validation Set: " + str(len(validationIndices)))
    logging.info("Done:\t Determine folds")


    logging.info("Start:\t Learning with " + CL_type + " and " + str(len(kFolds)) + " folds")
    extractionTime = time.time() - t_start

    classifierPackage = globals()[CL_type]  # Permet d'appeler un module avec une string
    classifierModule = getattr(classifierPackage, CL_type)
    classifierClass = getattr(classifierModule, CL_type)
    classifierGridSearch = getattr(classifierModule, "gridSearch_hdf5")
    analysisModule = getattr(classifierPackage, "analyzeResults")

    kFoldPredictedTrainLabels = []
    kFoldPredictedTestLabels = []
    kFoldPredictedValidationLabels = []
    kFoldLearningTime = []
    kFoldPredictionTime = []
    kFoldClassifier = []

    gridSearch=True
    if gridSearch:
        logging.info("Start:\t Randomsearching best settings for monoview classifiers")
        bestSettings, fusionConfig = classifierGridSearch(DATASET, classificationKWARGS, learningIndices
                                                          , metric=metrics[0], nIter=nIter)
        classificationKWARGS["classifiersConfigs"] = bestSettings
        try:
            classificationKWARGS["fusionMethodConfig"] = fusionConfig
        except:
            pass
        logging.info("Done:\t Randomsearching best settings for monoview classifiers")

    logging.info("Start:\t Classification")
    # Begin Classification
    for foldIdx, fold in enumerate(kFolds):
        if fold != range(classificationSetLength):
            fold.sort()
            logging.info("\tStart:\t Fold number " + str(foldIdx + 1))
            trainIndices = [index for index in range(datasetLength) if (index not in fold) and (index not in validationIndices)]
            DATASET_LENGTH = len(trainIndices)
            classifier = classifierClass(NB_VIEW, DATASET_LENGTH, DATASET.get("labels").value, NB_CORES=nbCores, **classificationKWARGS)

            classifier.fit_hdf5(DATASET, trainIndices=trainIndices)
            kFoldClassifier.append(classifier)

            learningTime = time.time() - extractionTime - t_start
            kFoldLearningTime.append(learningTime)
            logging.info("\tStart: \t Classification")
            kFoldPredictedTrainLabels.append(classifier.predict_hdf5(DATASET, usedIndices=trainIndices))
            kFoldPredictedTestLabels.append(classifier.predict_hdf5(DATASET, usedIndices=fold))
            kFoldPredictedValidationLabels.append(classifier.predict_hdf5(DATASET, usedIndices=validationIndices))

            kFoldPredictionTime.append(time.time() - extractionTime - t_start - learningTime)
            logging.info("\tDone: \t Fold number " + str(foldIdx + 1))

    classificationTime = time.time() - t_start

    logging.info("Done:\t Classification")
    logging.info("Info:\t Time for Classification: " + str(int(classificationTime)) + "[s]")
    logging.info("Start:\t Result Analysis for " + CL_type)

    times = (extractionTime, kFoldLearningTime, kFoldPredictionTime, classificationTime)

    stringAnalysis, imagesAnalysis, metricsScores = analysisModule.execute(kFoldClassifier, kFoldPredictedTrainLabels,
                                                            kFoldPredictedTestLabels, kFoldPredictedValidationLabels,
                                                            DATASET, classificationKWARGS, learningRate, LABELS_DICTIONARY,
                                                            views, nbCores, times, kFolds, name, nbFolds,
                                                            validationIndices, gridSearch, nIter, metrics)
    labelsSet = set(LABELS_DICTIONARY.values())
    logging.info(stringAnalysis)
    featureString = "-".join(views)
    labelsString = "-".join(labelsSet)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    CL_type_string = CL_type
    if CL_type=="Fusion":
        CL_type_string += "-"+classificationKWARGS["fusionType"]+"-"+classificationKWARGS["fusionMethod"]+"-"+"-".join(classificationKWARGS["classifiersNames"])
    elif CL_type=="Mumbo":
        CL_type_string += "-"+"-".join(classificationKWARGS["classifiersNames"])
    outputFileName = "Results/" + timestr + "Results-" + CL_type_string + "-" + featureString + '-' + labelsString + \
                     '-learnRate' + str(learningRate) + '-' + name

    outputTextFile = open(outputFileName + '.txt', 'w')
    outputTextFile.write(stringAnalysis)
    outputTextFile.close()

    if imagesAnalysis is not None:
        for imageName in imagesAnalysis:
            if os.path.isfile(outputFileName + imageName + ".png"):
                for i in range(1,20):
                    testFileName = outputFileName + imageName + "-" + str(i) + ".png"
                    if os.path.isfile(testFileName )!=True:
                        imagesAnalysis[imageName].savefig(testFileName)
                        break

            imagesAnalysis[imageName].savefig(outputFileName + imageName + '.png')

    logging.info("Done:\t Result Analysis")
    return CL_type, classificationKWARGS, metricsScores


if __name__=='__main__':

    # Argument Parser
    parser = argparse.ArgumentParser(
        description='This file is used to classifiy multiview data thanks to three methods : Fusion (early & late), Multiview Machines, Mumbo.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # create the top-level parser



    groupStandard = parser.add_argument_group('Standard arguments')
    groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
    groupStandard.add_argument('--name', metavar='STRING', action='store', help='Name of Database (default: %(default)s)',
                               default='Caltech')
    groupStandard.add_argument('--type', metavar='STRING', action='store', help='Type of database : .hdf5 or .csv',
                               default='.csv')
    groupStandard.add_argument('--views', metavar='STRING', action='store',
                               help='Name of the views selected for learning', default='RGB:HOG:SIFT')
    groupStandard.add_argument('--pathF', metavar='STRING', action='store',
                               help='Path to the views (default: %(default)s)',
                               default='../FeatExtraction/Results-FeatExtr/')

    groupClass = parser.add_argument_group('Classification arguments')
    groupClass.add_argument('--CL_split', metavar='FLOAT', action='store',
                            help='Determine the learning rate if > 1.0, number of fold for cross validation', type=float,
                            default=0.9)
    groupClass.add_argument('--CL_nbFolds', metavar='INT', action='store', help='Number of folds in cross validation',
                            type=int, default=3)
    groupClass.add_argument('--CL_nb_class', metavar='INT', action='store', help='Number of classes, -1 for all', type=int,
                            default=4)
    groupClass.add_argument('--CL_classes', metavar='STRING', action='store',
                            help='Classes used in the dataset (names of the folders) if not filled, random classes will be'
                                 ' selected ex. walrus:mole:leopard', default="")
    groupClass.add_argument('--CL_type', metavar='STRING', action='store',
                            help='Determine which multiview classifier to use', default='Mumbo')
    groupClass.add_argument('--CL_cores', metavar='INT', action='store', help='Number of cores, -1 for all', type=int,
                            default=1)

    groupMumbo = parser.add_argument_group('Mumbo arguments')
    groupMumbo.add_argument('--MU_type', metavar='STRING', action='store',
                            help='Determine which monoview classifier to use with Mumbo',
                            default='DecisionTree:DecisionTree:DecisionTree:DecisionTree')
    groupMumbo.add_argument('--MU_config', metavar='STRING', action='store', nargs='+',
                            help='Configuration for the monoview classifier in Mumbo', default=['1:0.02', '1:0.018',
                                                                                                '1:0.1',
                                                                                                '2:0.09'])
    groupMumbo.add_argument('--MU_iter', metavar='INT', action='store', nargs=3,
                            help='Max number of iteration, min number of iteration, convergeance threshold', type=float,
                            default=[1000,300,0.0005])

    groupFusion = parser.add_argument_group('Fusion', "poulet")
    groupFusion.add_argument('--FU_type', metavar='STRING', action='store',
                             help='Determine which type of fusion to use', default='LateFusion')
    groupFusion.add_argument('--FU_method', metavar='STRING', action='store',
                             help='Determine which method of fusion to use', default='WeightedLinear')
    groupFusion.add_argument('--FU_method_config', metavar='STRING', action='store', nargs='+',
                             help='Configuration for the fusion method', default=['1:1:1:1'])
    groupFusion.add_argument('--FU_cl_names', metavar='STRING', action='store',
                             help='Names of the monoview classifiers used',
                             default='RandomForest:SGD:SVC:DecisionTree')
    groupFusion.add_argument('--FU_cl_config', metavar='STRING', action='store', nargs='+',
                             help='Configuration for the monoview classifiers used', default=['3:4', 'log:l2', '10:linear',
                                                                                              '4'])
    print parser
    args = parser.parse_args()
    print args
    views = args.views.split(":")
    dataBaseType = args.type
    NB_VIEW = len(views)
    mumboClassifierConfig = [argument.split(':') for argument in args.MU_config]

    LEARNING_RATE = args.CL_split
    nbFolds = args.CL_nbFolds
    NB_CLASS = args.CL_nb_class
    LABELS_NAMES = args.CL_classes.split(":")
    mumboclassifierNames = args.MU_type.split(':')
    NB_CORES = args.CL_cores
    fusionClassifierNames = args.FU_cl_names.split(":")
    fusionClassifierConfig = [argument.split(':') for argument in args.FU_cl_config]
    fusionMethodConfig = [argument.split(':') for argument in args.FU_method_config]
    FusionKWARGS = {"fusionType":args.FU_type, "fusionMethod":args.FU_method,
                    "classifiersNames":fusionClassifierNames, "classifiersConfigs":fusionClassifierConfig,
                    'fusionMethodConfig':fusionMethodConfig}
    MumboKWARGS = {"classifiersConfigs":mumboClassifierConfig,
                   "classifiersNames":mumboclassifierNames, "maxIter":int(args.MU_iter[0]),
                   "minIter":int(args.MU_iter[1]), "threshold":args.MU_iter[2]}
    dir = os.path.dirname(os.path.abspath(__file__)) + "/Results/"
    logFileName = time.strftime("%Y%m%d-%H%M%S") + "-CMultiV-" + args.CL_type + "-" + "_".join(views) + "-" + args.name + \
                  "-LOG"
    logFile = dir + logFileName
    if os.path.isfile(logFile + ".log"):
        for i in range(1, 20):
            testFileName = logFileName + "-" + str(i) + ".log"
            if not (os.path.isfile(dir + testFileName)):
                logfile = dir + testFileName
                break
    else:
        logFile += ".log"
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=logFile, level=logging.DEBUG,
                        filemode='w')
    if args.log:
        logging.getLogger().addHandler(logging.StreamHandler())
    arguments = {"CL_type": args.CL_type,
                 "views": args.views.split(":"),
                 "NB_VIEW": len(args.views.split(":")),
                 "NB_CLASS": len(args.CL_classes.split(":")),
                 "LABELS_NAMES": args.CL_classes.split(":"),
                 "FusionKWARGS": FusionKWARGS,
                 "MumboKWARGS": MumboKWARGS}

    logging.info("Start:\t Read " + str.upper(args.type[1:]) + " Database Files for " + args.name)

    getDatabase = getattr(DB, "get" + args.name + "DB" + args.type[1:])
    DATASET, LABELS_DICTIONARY = getDatabase(views, args.pathF, args.name, NB_CLASS, LABELS_NAMES)

    logging.info("Info:\t Labels used: " + ", ".join(LABELS_DICTIONARY.values()))
    logging.info("Info:\t Length of dataset:" + str(DATASET.get("Metadata").attrs["datasetLength"]))

    ExecMultiview(DATASET, args.name, args.CL_split, args.CL_nbFolds, args.CL_cores, args.type, args.pathF,
                  LABELS_DICTIONARY, gridSearch=False, **arguments)



    # # Stats Result
# y_test_pred = cl_res.predict(X_test)
# classLabelsDesc = pd.read_csv(args.pathF + args.fileCLD, sep=";", names=['label', 'name'])
# classLabelsNames = classLabelsDesc.name
# #logging.info("" + str(classLabelsNames))
# classLabelsNamesList = classLabelsNames.values.tolist()
# #logging.info(""+ str(classLabelsNamesList))
#
# logging.info("Start:\t Statistic Results")
#
# #Accuracy classification score
# accuracy_score = ExportResults.accuracy_score(y_test, y_test_pred)
#
# # Classification Report with Precision, Recall, F1 , Support
# logging.info("Info:\t Classification report:")
# filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-Report"
# logging.info("\n" + str(metrics.classification_report(y_test, y_test_pred, labels = range(0,len(classLabelsDesc.name)), target_names=classLabelsNamesList)))
# scores_df = ExportResults.classification_report_df(directory, filename, y_test, y_test_pred, range(0, len(classLabelsDesc.name)), classLabelsNamesList)
#
# # Create some useful statistcs
# logging.info("Info:\t Statistics:")
# filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-Stats"
# stats_df = ExportResults.classification_stats(directory, filename, scores_df, accuracy_score)
# logging.info("\n" + stats_df.to_string())
#
# # Confusion Matrix
# logging.info("Info:\t Calculate Confusionmatrix")
# filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-ConfMatrix"
# df_conf_norm = ExportResults.confusion_matrix_df(directory, filename, y_test, y_test_pred, classLabelsNamesList)
# filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-ConfMatrixImg"
# ExportResults.plot_confusion_matrix(directory, filename, df_conf_norm)
#
# logging.info("Done:\t Statistic Results")
#
#
# # Plot Result
# logging.info("Start:\t Plot Result")
# np_score = ExportResults.calcScorePerClass(y_test, cl_res.predict(X_test).astype(int))
# ### directory and filename the same as CSV Export
# filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-Score"
# ExportResults.showResults(directory, filename, args.name, args.feat, np_score)
# logging.info("Done:\t Plot Result")


#
# NB_CLASS = 5
# NB_ITER = 100
# classifierName="DecisionTree"
# NB_CORES = 3
# pathToAwa = "/home/doob/"
# views = ['phog-hist', 'decaf', 'cq-hist']
# NB_VIEW = len(views)
# LEARNING_RATE = 1.0
#
# print "Getting db ..."
# DATASET, LABELS, viewDictionnary, labelDictionnary = DB.getAwaData(pathToAwa, NB_CLASS, views)
# target_names = [labelDictionnary[label] for label in labelDictionnary]
# # DATASET, LABELS = DB.getDbfromCSV('/home/doob/OriginalData/')
# # NB_VIEW = 3
# LABELS = np.array([int(label) for label in LABELS])
# # print target_names
# # print labelDictionnary
# DATASET_LENGTH = len(LABELS)
#
# DATASET_LENGTH = len(trainLabels)
# # print len(trainData), trainData[0].shape, len(trainLabels)
# print "Done."
#
# print 'Training Mumbo ...'
# # DATASET, VIEW_DIMENSIONS, LABELS = DB.createFakeData(NB_VIEW, DATASET_LENGTH, NB_CLASS)
# print "Trained."
#
# print "Predicting ..."
# predictedTrainLabels = Mumbo.classifyMumbo(trainData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
# predictedTestLabels = Mumbo.classifyMumbo(testData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
# print 'Done.'
# print 'Reporting ...'
# predictedTrainLabelsByIter = Mumbo.classifyMumbobyIter(trainData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
# predictedTestLabelsByIter = Mumbo.classifyMumbobyIter(testData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
# print str(NB_VIEW)+" views, "+str(NB_CLASS)+" classes, "+str(classifierConfig)+" depth trees"
# print "Best views = "+str(bestViews)
# print "Is equal : "+str((predictedTrainLabels==predictedTrainLabelsByIter[NB_ITER-1]).all())
#
# print "On train : "
# print classification_report(trainLabels, predictedTrainLabels, target_names=target_names)
# print "On test : "
# print classification_report(testLabels, predictedTestLabels, target_names=target_names)

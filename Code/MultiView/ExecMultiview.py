import sys
import os.path
sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from MultiView import *

import GetMutliviewDb as DB
import argparse
import numpy as np
import datetime
import os
import logging
import time


# Argument Parser
parser = argparse.ArgumentParser(
        description='This file is used to classifiy multiview data thanks to three methods : Fusion (early & late), Multiview Machines, Mumbo.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

groupStandard = parser.add_argument_group('Standard arguments')
groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
groupStandard.add_argument('--name', metavar='STRING', action='store', help='Name of Database (default: %(default)s)',
                           default='Caltech')
groupStandard.add_argument('--features', metavar='STRING', action='store',
                           help='Name of the features selected for learning', default='RGB:HOG:SIFT')
groupStandard.add_argument('--pathF', metavar='STRING', action='store',
                           help='Path to the features (default: %(default)s)',
                           default='../FeatExtraction/Results-FeatExtr/')

groupClass = parser.add_argument_group('Classification arguments')
groupClass.add_argument('--CL_split', metavar='FLOAT', action='store', help='Determine the learning rate if > 1.0, number of fold for cross validation', type=float,
                        default=1.0)
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
                        help='Determine which monoview classifier to use with Mumbo', default='DecisionTree:DecisionTree:DecisionTree')
groupMumbo.add_argument('--MU_config', metavar='STRING', action='store', nargs='+',
                        help='Configuration for the monoview classifier in Mumbo', default='3:1.0 3:1.0 3:1.0')
groupMumbo.add_argument('--MU_iter', metavar='INT', action='store',
                        help='Number of iterations in Mumbos learning process', type=int, default=5)

groupFusion = parser.add_argument_group('Fusion arguments')
groupFusion.add_argument('--FU_cl_type', metavar='STRING', action='store',
                         help='Determine which monoview classifier to use with fusion', default='RandomForest')
groupFusion.add_argument('--FU_type', metavar='STRING', action='store',
                         help='Determine which type of fusion to use', default='EarlyFusion')
groupFusion.add_argument('--FU_method', metavar='STRING', action='store',
                         help='Determine which method of fusion to use', default='linearWeighted')
groupFusion.add_argument('--FU_config', metavar='STRING', action='store',
                         help='Configuration for the fusion method', default='1.0:1.0:1.0')
groupFusion.add_argument('--FU_cl_config', metavar='STRING', action='store',
                         help='Configuration for the monoview classifier', default='100:10:5')

args = parser.parse_args()
features = args.features.split(":")
NB_VIEW = len(features)
mumboClassifierConfig = [argument.split(':') for argument in args.MU_config]

LEARNING_RATE = args.CL_split
NB_CLASS = args.CL_nb_class
LABELS_NAMES = args.CL_classes.split(":")
classifierNames = args.MU_type.split(':')
NB_ITER = args.MU_iter
NB_CORES = args.CL_cores
fusionClassifierConfig = args.FU_cl_config.split(":")
fusionMethodConfig = args.FU_config.split(":")
FusionArguments = (args.FU_type, args.FU_method, fusionMethodConfig, args.FU_cl_type, fusionClassifierConfig)
MumboArguments = (mumboClassifierConfig, NB_ITER, classifierNames)


dir = os.path.dirname(os.path.abspath(__file__)) + "/Results/"
logFileName = datetime.datetime.now().strftime(
    "%Y_%m_%d") + "-CMultiV-" + args.CL_type + "-" + "_".join(features) + "-" + args.name + "-LOG"
logFile = dir + logFileName

if os.path.isfile(logFile + ".log"):
    for i in range(1, 20):
        testFileName = logFileName + "-" + str(i) + ".log"
        if not (os.path.isfile(dir + testFileName)):
            logfile = dir + testFileName
            break
else:
    logFile = logFile + ".log"

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=logFile, level=logging.DEBUG,
                    filemode='w')

if (args.log):
    logging.getLogger().addHandler(logging.StreamHandler())

# Determine the Database to extract features
logging.info("### Main Programm for Multiview Classification")
logging.info("### Classification - Database : " + str(args.name) + " ; Views : " + ", ".join(features) +
              " ; Algorithm : " + args.CL_type + " ; Cores : " + str(NB_CORES))
t_start = time.time()


# Read the features
logging.info("Start:\t Read CSV Database Files for "+args.name)

getDatabase = getattr(DB, "get"+args.name+"DB")

DATASET, CLASS_LABELS, LABELS_DICTIONARY, datasetLength = getDatabase(features, args.pathF, args.name, NB_CLASS, LABELS_NAMES)

labelsSet = LABELS_DICTIONARY.values()
logging.info("Info:\t Labels used: "+", ".join(labelsSet))
logging.info("Info:\t Length of dataset:" + str(datasetLength))

for viewIndice in range(NB_VIEW):
    logging.info("Info:\t Shape of " + features[viewIndice] + " :" + str(
        DATASET[viewIndice].shape))
logging.info("Done:\t Read CSV Files")


# Calculate Train/Test data
if LEARNING_RATE<=1.0:
    logging.info("Start:\t Determine Train/Test split for ratio " + str(LEARNING_RATE))
else:
    logging.info("Start:\t Determine "+str(LEARNING_RATE)+" folds for cross validation ")


if LEARNING_RATE <=1.0:
    # trainData, trainLabels, testData, testLabels = DB.extractRandomTrainingSet(DATASET, CLASS_LABELS, LEARNING_RATE,
    #                                                                        datasetLength, NB_VIEW, NB_CLASS)
    kFolds = DB.extractRandomTrainingSet(DATASET, CLASS_LABELS, LEARNING_RATE, datasetLength, NB_VIEW, NB_CLASS)

elif LEARNING_RATE == 1.0:
    kFolds = [range(datasetLength), []]

else:
    kFolds = DB.getKFoldIndices(LEARNING_RATE, CLASS_LABELS, datasetLength, NB_CLASS)


logging.info("Info:\t Length of Learning Set: " + str(len(kFolds[0])))
logging.info("Info:\t Length of Testing Set: " + str(datasetLength-len(kFolds[0])))
logging.info("Done:\t Determine folds")
extractionTime = time.time() - t_start

classifierPackage = globals()[args.CL_type]  # Permet d'appeler un module avec une string
trainArguments = globals()[args.CL_type+'Arguments']
classifierModule = getattr(classifierPackage, args.CL_type)
train = getattr(classifierModule, "train")
predict = getattr(classifierModule, "predict")
analysisModule = getattr(classifierPackage, "analyzeResults")

kFoldPredictedTrainLabels = []
kFoldPredictedTestLabels = []
kFoldLearningTime = []
kFoldPredictionTime = []
kFoldClassifier = []

# Begin Classification
logging.info("Start:\t Learning with " + args.CL_type + " and "+ str(len(kFolds))+" folds")
for foldIdx, fold in enumerate(kFolds) :
    if fold:
        logging.info("\tStart:\t Fold number "+str(foldIdx+1))
        trainIndices = [index for index in range(datasetLength) if index not in fold]
        testData = dict((key, value[fold, :]) for key, value in DATASET.iteritems())
        trainData = dict((key, value[trainIndices, :]) for key, value in DATASET.iteritems())
        testLabels = CLASS_LABELS[fold]
        trainLabels = CLASS_LABELS[trainIndices]
        DATASET_LENGTH = len(trainLabels)
        classifier = train(trainData, trainLabels, DATASET_LENGTH, NB_VIEW, NB_CLASS, NB_CORES, trainArguments)
        kFoldClassifier.append(classifier)

        learningTime = time.time() - extractionTime - t_start
        kFoldLearningTime.append(learningTime)
        logging.info("\tStart: \t Classification")

        kFoldPredictedTrainLabels.append(predict(trainData, classifier, NB_CLASS))
        kFoldPredictedTestLabels.append(predict(testData, classifier, NB_CLASS))

        kFoldPredictionTime.append(time.time() - extractionTime - t_start - learningTime)
        logging.info("\tDone: \t Fold number "+str(foldIdx+1))


classificationTime = time.time() - t_start

logging.info("Done:\t Classification")
logging.info("Info:\t Time for Classification: " + str(int(classificationTime)) + "[s]")
logging.info("Start:\t Result Analysis for " + args.CL_type)

times = (extractionTime, kFoldLearningTime, kFoldPredictionTime, classificationTime)

stringAnalysis, imagesAnalysis = analysisModule.execute(kFoldClassifier, kFoldPredictedTrainLabels,
                                                        kFoldPredictedTestLabels, DATASET, CLASS_LABELS,
                                                        NB_CLASS, trainArguments, LEARNING_RATE, LABELS_DICTIONARY,
                                                        features, NB_CORES, times, NB_VIEW, kFolds, args.name)
logging.info(stringAnalysis)
featureString = "-".join(features)
labelsString = "-".join(labelsSet)
timestr = time.strftime("%Y%m%d-%H%M%S")
outputFileName = "Results/"+timestr+"Results-"+args.CL_type+"-" + ":".join(classifierNames) + '-' + featureString + '-' + labelsString + '-learnRate' + str(
    LEARNING_RATE) + '-nbIter' + str(NB_ITER) + '-' + args.name

outputTextFile = open(outputFileName + '.txt', 'w')
outputTextFile.write(stringAnalysis)
outputTextFile.close()

if imagesAnalysis is not None:
    for imageName in imagesAnalysis:
        # if os.path.isfile(outputFileName + imageName + ".png"):
        #     for i in range(1,20):
        #         testFileName = outputFileName + imageName + "-" + str(i) + ".png"
        #         if os.path.isfile(testFileName )!=True:
        #             imagesAnalysis[imageName].savefig(testFileName)
        #             break

        imagesAnalysis[imageName].savefig(outputFileName + imageName + '.png')

logging.info("Done:\t Result Analysis")






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
# scores_df = ExportResults.classification_report_df(dir, filename, y_test, y_test_pred, range(0, len(classLabelsDesc.name)), classLabelsNamesList)
#
# # Create some useful statistcs
# logging.info("Info:\t Statistics:")
# filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-Stats"
# stats_df = ExportResults.classification_stats(dir, filename, scores_df, accuracy_score)
# logging.info("\n" + stats_df.to_string())
#
# # Confusion Matrix
# logging.info("Info:\t Calculate Confusionmatrix")
# filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-ConfMatrix"
# df_conf_norm = ExportResults.confusion_matrix_df(dir, filename, y_test, y_test_pred, classLabelsNamesList)
# filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-ConfMatrixImg"
# ExportResults.plot_confusion_matrix(dir, filename, df_conf_norm)
#
# logging.info("Done:\t Statistic Results")
#
#
# # Plot Result
# logging.info("Start:\t Plot Result")
# np_score = ExportResults.calcScorePerClass(y_test, cl_res.predict(X_test).astype(int))
# ### dir and filename the same as CSV Export
# filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-Score"
# ExportResults.showResults(dir, filename, args.name, args.feat, np_score)
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
# DATASET, CLASS_LABELS, viewDictionnary, labelDictionnary = DB.getAwaData(pathToAwa, NB_CLASS, views)
# target_names = [labelDictionnary[label] for label in labelDictionnary]
# # DATASET, CLASS_LABELS = DB.getDbfromCSV('/home/doob/OriginalData/')
# # NB_VIEW = 3
# CLASS_LABELS = np.array([int(label) for label in CLASS_LABELS])
# # print target_names
# # print labelDictionnary
# DATASET_LENGTH = len(CLASS_LABELS)
#
# DATASET_LENGTH = len(trainLabels)
# # print len(trainData), trainData[0].shape, len(trainLabels)
# print "Done."
#
# print 'Training Mumbo ...'
# # DATASET, VIEW_DIMENSIONS, CLASS_LABELS = DB.createFakeData(NB_VIEW, DATASET_LENGTH, NB_CLASS)
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

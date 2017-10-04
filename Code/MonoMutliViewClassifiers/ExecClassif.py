# Import built-in modules
import argparse
import pkgutil  # for TimeStamp in CSVFile
import os
import time
import itertools
import sys
import select
import logging
import errno

# Import 3rd party modules
from joblib import Parallel, delayed
import numpy as np
import math
import matplotlib

# Import own modules
import Multiview
import Metrics
import MonoviewClassifiers
from Multiview.ExecMultiview import ExecMultiview, ExecMultiview_multicore
from Monoview.ExecClassifMonoView import ExecMonoview, ExecMonoview_multicore
import Multiview.GetMultiviewDb as DB
from Versions import testVersions
from ResultAnalysis import resultAnalysis, analyzeLabels

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

matplotlib.use('Agg')  # Anti-Grain Geometry C++ library to make a raster (pixel) image of the figure


def initLogFile(args):
    resultDirectory = "../../Results/" + args.name + "/started_" + time.strftime("%Y_%m_%d-%H_%M") + "/"
    logFileName = time.strftime("%Y%m%d-%H%M%S") + "-CMultiV-" + args.CL_type + "-" + "_".join(
        args.views) + "-" + args.name + \
                  "-LOG"
    if not os.path.exists(os.path.dirname(resultDirectory + logFileName)):
        try:
            os.makedirs(os.path.dirname(resultDirectory + logFileName))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    logFile = resultDirectory + logFileName
    if os.path.isfile(logFile + ".log"):
        for i in range(1, 20):
            testFileName = logFileName + "-" + str(i) + ".log"
            if not (os.path.isfile(resultDirectory + testFileName)):
                logfile = resultDirectory + testFileName
                break
    else:
        logFile += ".log"
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=logFile, level=logging.DEBUG,
                        filemode='w')
    if args.log:
        logging.getLogger().addHandler(logging.StreamHandler())

    return resultDirectory


def input(timeout=15):
    print "You have " + str(timeout) + " seconds to stop the script by typing n"

    i, o, e = select.select([sys.stdin], [], [], timeout)

    if i:
        return sys.stdin.readline().strip()
    else:
        return "y"


def confirm(resp=True, timeout=15):
    ans = input(timeout)
    if not ans:
        return resp
    if ans not in ['y', 'Y', 'n', 'N']:
        print 'please enter y or n.'
    if ans == 'y' or ans == 'Y':
        return True
    if ans == 'n' or ans == 'N':
        return False


def initMultipleDatasets(args, nbCores):
    """Used to create copies of the dataset if multicore computation is used
    Needs arg.pathF and arg.name"""
    if nbCores > 1:
        if DB.datasetsAlreadyExist(args.pathF, args.name, nbCores):
            logging.debug("Info:\t Enough copies of the dataset are already available")
            pass
        else:
            logging.debug("Start:\t Creating " + str(nbCores) + " temporary datasets for multiprocessing")
            logging.warning(" WARNING : /!\ This may use a lot of HDD storage space : " +
                            str(os.path.getsize(args.pathF + args.name + ".hdf5") * nbCores / float(
                                1024) / 1000 / 1000) + " Gbytes /!\ ")
            confirmation = confirm()
            if not confirmation:
                sys.exit(0)
            else:
                datasetFiles = DB.copyHDF5(args.pathF, args.name, nbCores)
                logging.debug("Start:\t Creating datasets for multiprocessing")
                return datasetFiles


def initViews(DATASET, args):
    """Used to return the views names that will be used by the algos, their indices and all the views names
    Needs args.views"""
    NB_VIEW = DATASET.get("Metadata").attrs["nbView"]
    if args.views != [""]:
        allowedViews = args.views
        allViews = [str(DATASET.get("View" + str(viewIndex)).attrs["name"]) for viewIndex in range(NB_VIEW)]
        views = [str(DATASET.get("View" + str(viewIndex)).attrs["name"]) for viewIndex in range(NB_VIEW) if
                 str(DATASET.get("View" + str(viewIndex)).attrs["name"]) in allowedViews]
        viewsIndices = [viewIndex for viewIndex in range(NB_VIEW) if
                        str(DATASET.get("View" + str(viewIndex)).attrs["name"]) in allowedViews]
        return views, viewsIndices, allViews
    else:
        views = [str(DATASET.get("View" + str(viewIndex)).attrs["name"]) for viewIndex in range(NB_VIEW)]
        viewsIndices = np.arange(NB_VIEW)
        allViews = views
        return views, viewsIndices, allViews


def initBenchmark(args):
    """Used to create a list of all the algorithm packages names used for the benchmark
    Needs args.CL_type, args.CL_algos_multiview, args.MU_types, args.FU_types, args.FU_late_methods,
    args.FU_early_methods, args.CL_algos_monoview"""
    benchmark = {"Monoview": {}, "Multiview": []}
    if args.CL_type.split(":") == ["Benchmark"]:
        # if args.CL_algorithm=='':
        fusionModulesNames = [name for _, name, isPackage
                              in pkgutil.iter_modules(['Multiview/Fusion/Methods']) if not isPackage]
        fusionModules = [getattr(Multiview.Fusion.Methods, fusionModulesName)
                         for fusionModulesName in fusionModulesNames]
        fusionClasses = [getattr(fusionModule, fusionModulesName + "Classifier")
                         for fusionModulesName, fusionModule in zip(fusionModulesNames, fusionModules)]
        fusionMethods = dict((fusionModulesName, [name for _, name, isPackage in
                                                  pkgutil.iter_modules(
                                                      ["Multiview/Fusion/Methods/" + fusionModulesName + "Package"])
                                                  if not isPackage])
                             for fusionModulesName, fusionClasse in zip(fusionModulesNames, fusionClasses))
        allMonoviewAlgos = [name for _, name, isPackage in
                            pkgutil.iter_modules(['MonoviewClassifiers'])
                            if (not isPackage)]
        fusionMonoviewClassifiers = allMonoviewAlgos
        allFusionAlgos = {"Methods": fusionMethods, "Classifiers": fusionMonoviewClassifiers}
        allMumboAlgos = [name for _, name, isPackage in
                         pkgutil.iter_modules(['Multiview/Mumbo/Classifiers'])
                         if not isPackage and not name in ["SubSampling", "ModifiedMulticlass", "Kover"]]
        allMultiviewAlgos = {"Fusion": allFusionAlgos, "Mumbo": allMumboAlgos}
        benchmark = {"Monoview": allMonoviewAlgos, "Multiview": allMultiviewAlgos}

    if "Multiview" in args.CL_type.strip(":"):
        benchmark["Multiview"] = {}
        if args.CL_algos_multiview == [""]:
            algosMutliview = ["Mumbo", "Fusion"]
        else:
            algosMutliview = args.CL_algos_multiview
        if "Mumbo" in algosMutliview:
            benchmark["Multiview"]["Mumbo"] = args.MU_types
        if "Fusion" in algosMutliview:
            benchmark["Multiview"]["Fusion"] = {}
            benchmark["Multiview"]["Fusion"]["Methods"] = dict(
                (fusionType, []) for fusionType in args.FU_types)
            if "LateFusion" in args.FU_types:
                if args.FU_late_methods== [""]:
                    benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"] = [name for _, name, isPackage in
                                                                                 pkgutil.iter_modules([
                                                                                     "Multiview/Fusion/Methods/LateFusionPackage"])
                                                                                 if not isPackage]
                else:
                    benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"] = args.FU_late_methods
            if "EarlyFusion" in args.FU_types:
                if args.FU_early_methods == [""]:
                    benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] = [name for _, name, isPackage in
                                                                                  pkgutil.iter_modules([
                                                                                      "Multiview/Fusion/Methods/EarlyFusionPackage"])
                                                                                  if not isPackage]
                else:
                    benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] = args.FU_early_methods
            if args.CL_algos_monoview == ['']:
                benchmark["Multiview"]["Fusion"]["Classifiers"] = [name for _, name, isPackage in
                                                                   pkgutil.iter_modules(['MonoviewClassifiers'])
                                                                   if (not isPackage) and (name != "SGD") and (
                                                                       name[:3] != "SVM")
                                                                   and (name != "SCM")]
            else:
                benchmark["Multiview"]["Fusion"]["Classifiers"] = args.CL_algos_monoview

    if "Monoview" in args.CL_type.strip(":"):
        if args.CL_algos_monoview == ['']:
            benchmark["Monoview"] = [name for _, name, isPackage in pkgutil.iter_modules(["MonoviewClassifiers"])
                                      if not isPackage]

        else:
            benchmark["Monoview"] = args.CL_algos_monoview
    return benchmark


def initMonoviewArguments(benchmark, argumentDictionaries, views, allViews, DATASET, NB_CLASS, kwargsInit):
    if benchmark["Monoview"]:
        argumentDictionaries["Monoview"] = []
        for view in views:
            for classifier in benchmark["Monoview"]:
                if classifier == "SCM":
                    if DATASET.get("View" + str(allViews.index(view))).attrs["binary"]:
                        arguments = {
                            "args": {classifier + "KWARGS": kwargsInit[classifier + "KWARGSInit"], "feat": view,
                                     "CL_type": classifier, "nbClass": NB_CLASS}, "viewIndex": allViews.index(view)}
                        argumentDictionaries["Monoview"].append(arguments)
                    else:
                        pass
                else:
                    arguments = {
                        "args": {classifier + "KWARGS": kwargsInit[classifier + "KWARGSInit"], "feat": view,
                                 "CL_type": classifier, "nbClass": NB_CLASS}, "viewIndex": allViews.index(view)}
                    argumentDictionaries["Monoview"].append(arguments)
    return argumentDictionaries


def initMonoviewKWARGS(args, classifiersNames):
    monoviewKWARGS = {}
    for classifiersName in classifiersNames:
        classifierModule = getattr(MonoviewClassifiers, classifiersName)
        monoviewKWARGS[classifiersName+"KWARGSInit"] = classifierModule.getKWARGS([(key, value) for key, value in vars(args).iteritems() if key.startswith("CL_"+classifiersName)])
    return monoviewKWARGS


def initKWARGS(args, benchmark):
    if "Monoview" in benchmark:
        monoviewKWARGS = initMonoviewKWARGS(args, benchmark["Monoview"])



    # kwargsInit = {
    #     "RandomForestKWARGSInit": {"0": map(int, args.CL_RF_trees.split())[0],
    #                                "1": map(int, args.CL_RF_max_depth.split(":"))[0]},
    #     "SVMLinearKWARGSInit": {"0": map(int, args.CL_SVML_C.split(":"))[0]},
    #     "SVMRBFKWARGSInit": {"0": map(int, args.CL_SVMR_C.split(":"))[0]},
    #     "SVMPolyKWARGSInit": {"0": map(int, args.CL_SVMP_C.split(":"))[0],
    #                           '1': map(int, args.CL_SVMP_deg.split(":"))[0]},
    #     "DecisionTreeKWARGSInit": {"0": map(int, args.CL_DT_depth.split(":"))[0]},
    #     "SGDKWARGSInit": {"2": map(float, args.CL_SGD_alpha.split(":"))[0], "1": args.CL_SGD_penalty.split(":")[0],
    #                       "0": args.CL_SGD_loss.split(":")[0]},
    #     "KNNKWARGSInit": {"0": map(float, args.CL_KNN_neigh.split(":"))[0]},
    #     "AdaboostKWARGSInit": {"0": args.CL_Ada_n_est.split(":")[0], "1": args.CL_Ada_b_est.split(":")[0]},
    #     "SCMKWARGSInit": {"0": args.CL_SCM_max_rules.split(":")[0]},
    # }
    return monoviewKWARGS


def lateFusionSetArgs(views, viewsIndices, classes, method,
                      classifiersNames, classifiersConfig, fusionMethodConfig):
    arguments = {"CL_type": "Fusion",
                 "views": views,
                 "NB_VIEW": len(views),
                 "viewsIndices": viewsIndices,
                 "NB_CLASS": len(classes),
                 "LABELS_NAMES": args.CL_classes,
                 "FusionKWARGS": {"fusionType": "LateFusion", "fusionMethod": method,
                                  "classifiersNames": classifiersNames,
                                  "classifiersConfigs": classifiersConfig,
                                  'fusionMethodConfig': fusionMethodConfig,
                                  "nbView": (len(viewsIndices))}}
    return arguments


def initMultiviewArguments(args, benchmark, views, viewsIndices, scores, classifiersConfigs, classifiersNames,
                           NB_VIEW, metrics, argumentDictionaries):
    # metricModule = getattr(Metrics, metrics[0])
    multiviewArguments = []
    if "Multiview" in benchmark:
        for multiviewAlgoName in benchmark["Multiview"]:
            multiviewPackage = getattr(Multiview, multiviewAlgoName)
            mutliviewModule = getattr(multiviewPackage, multiviewAlgoName)
            multiviewArguments+= mutliviewModule.getArgs(args, benchmark, views, viewsIndices)
    # if benchmark["Multiview"]:
    #     for multiviewAlgoName in benchmark["Multiview"]:
    #         multiviewPackage = getattr(Multiview, multiviewAlgoName)
    #         multiviewArguments[]
    #     if "Fusion" in benchmark["Multiview"]:
    #         for method in benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"]:
    #             import pdb; pdb.set_trace()
    #         if args.FU_cl_names != ['']:
    #             monoClassifiers = args.FU_cl_names
    #             monoClassifiersConfigs = [globals()[classifier + "KWARGS"] for classifier in monoClassifiers]
    #             if args.FU_method_config != [""]:
    #                 fusionMethodConfigs = [map(float, config.split(":")) for config in args.FU_method_config]
    #             elif not hyperParamSearch:
    #                 raise ValueError("No config for fusion method given and no gridearch wanted")
    #             else:
    #                 try:
    #                     fusionMethodConfigs = [["config"] for method in
    #                                            benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"]]
    #                 except:
    #                     pass
    #             try:
    #                 for methodIndex, method in enumerate(benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"]):
    #                     if args.FU_fixed:
    #                         arguments = lateFusionSetArgs(views, viewsIndices, args.CL_classes, method,
    #                                                       args.FU_cl_names, monoClassifiersConfigs,
    #                                                       fusionMethodConfigs[methodIndex])
    #                         argumentDictionaries["Multiview"].append(arguments)
    #                     else:
    #                         for combination in itertools.combinations_with_replacement(range(len(monoClassifiers)),
    #                                                                                    NB_VIEW):
    #                             monoClassifiersNamesComb = [monoClassifiers[index] for index in combination]
    #                             monoClassifiersConfigsComb = [monoClassifiersConfigs[index] for index in
    #                                                           combination]
    #                             arguments = lateFusionSetArgs(views, viewsIndices, args.CL_classes, method,
    #                                                           monoClassifiersNamesComb, monoClassifiersConfigsComb,
    #                                                           fusionMethodConfigs[methodIndex])
    #                             argumentDictionaries["Multiview"].append(arguments)
    #             except:
    #                 pass
    #         else:
    #             if "LateFusion" in benchmark["Multiview"]["Fusion"]["Methods"] and \
    #                             "Classifiers" in benchmark["Multiview"]["Fusion"]:
    #                 bestClassifiers = []
    #                 bestClassifiersConfigs = []
    #                 if argumentDictionaries["Monoview"] != {}:
    #                     for viewIndex, view in enumerate(views):
    #                         if metricModule.getConfig()[-14] == "h":
    #                             bestClassifiers.append(
    #                                 classifiersNames[viewIndex][np.argmax(np.array(scores[viewIndex]))])
    #                             bestClassifiersConfigs.append(
    #                                 classifiersConfigs[viewIndex][np.argmax(np.array(scores[viewIndex]))])
    #                         else:
    #                             bestClassifiers.append(
    #                                 classifiersNames[viewIndex][np.argmin(np.array(scores[viewIndex]))])
    #                             bestClassifiersConfigs.append(
    #                                 classifiersConfigs[viewIndex][np.argmin(np.array(scores[viewIndex]))])
    #                 else:
    #                     raise AttributeError("No Monoview classifiers asked in args and no monoview benchmark done.")
    #                 for method in benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"]:
    #                     arguments = lateFusionSetArgs(views, viewsIndices, args.CL_classes, method,
    #                                                   bestClassifiers, bestClassifiersConfigs,
    #                                                   fusionMethodConfig)
    #                     argumentDictionaries["Multiview"].append(arguments)
    #         if "EarlyFusion" in benchmark["Multiview"]["Fusion"]["Methods"] and \
    #                         "Classifiers" in benchmark["Multiview"]["Fusion"]:
    #             for method in benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"]:
    #                 for classifier in benchmark["Multiview"]["Fusion"]["Classifiers"]:
    #                     arguments = {"CL_type": "Fusion",
    #                                  "views": views,
    #                                  "NB_VIEW": len(views),
    #                                  "viewsIndices": viewsIndices,
    #                                  "NB_CLASS": len(args.CL_classes),
    #                                  "LABELS_NAMES": args.CL_classes,
    #                                  "FusionKWARGS": {"fusionType": "EarlyFusion", "fusionMethod": method,
    #                                                   "classifiersNames": [classifier],
    #                                                   "classifiersConfigs": [
    #                                                       initKWARGS[classifier + "KWARGSInit"]],
    #                                                   'fusionMethodConfig': fusionMethodConfig,
    #                                                   "nbView": (len(viewsIndices))}}
    #                     argumentDictionaries["Multiview"].append(arguments)
    #     if "Mumbo" in benchmark["Multiview"]:
    #         for combination in itertools.combinations_with_replacement(range(len(benchmark["Multiview"]["Mumbo"])),
    #                                                                    NB_VIEW):
    #             mumboClassifiersNames = [benchmark["Multiview"]["Mumbo"][index] for index in combination]
    #             arguments = {"CL_type": "Mumbo",
    #                          "views": views,
    #                          "NB_VIEW": len(views),
    #                          "viewsIndices": viewsIndices,
    #                          "NB_CLASS": len(args.CL_classes),
    #                          "LABELS_NAMES": args.CL_classes,
    #                          "MumboKWARGS": {"classifiersNames": mumboClassifiersNames,
    #                                          "maxIter": int(args.MU_iter[0]), "minIter": int(args.MU_iter[1]),
    #                                          "threshold": args.MU_iter[2],
    #                                          "classifiersConfigs": [argument.split(":") for argument in
    #                                                                 args.MU_config], "nbView": (len(viewsIndices))}}
    #             argumentDictionaries["Multiview"].append(arguments)
    argumentDictionaries["Multiview"] = multiviewArguments
    return argumentDictionaries


def arangeMetrics(metrics, metricPrinc):
    if [metricPrinc] in metrics:
        metricIndex = metrics.index([metricPrinc])
        firstMetric = metrics[0]
        metrics[0]=[metricPrinc]
        metrics[metricIndex]=firstMetric
    else:
        raise AttributeError(metricPrinc+" not in metric pool")
    return metrics


testVersions()
parser = argparse.ArgumentParser(
    description='This file is used to benchmark the scores fo multiple classification algorithm on multiview data.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

groupStandard = parser.add_argument_group('Standard arguments')
groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
groupStandard.add_argument('--name', metavar='STRING', action='store', help='Name of Database (default: %(default)s)',
                           default='Plausible')
groupStandard.add_argument('--type', metavar='STRING', action='store',
                           help='Type of database : .hdf5 or .csv (default: %(default)s)',
                           default='.hdf5')
groupStandard.add_argument('--views', metavar='STRING', action='store', nargs="+",
                           help='Name of the views selected for learning (default: %(default)s)',
                           default=[''])
groupStandard.add_argument('--pathF', metavar='STRING', action='store', help='Path to the views (default: %(default)s)',
                           default='/home/bbauvin/Documents/Data/Data_multi_omics/')
groupStandard.add_argument('--nice', metavar='INT', action='store', type=int,
                           help='Niceness for the process', default=0)

groupClass = parser.add_argument_group('Classification arguments')
groupClass.add_argument('--CL_split', metavar='FLOAT', action='store',
                        help='Determine the split between learning and validation sets', type=float,
                        default=0.7)
groupClass.add_argument('--CL_nbFolds', metavar='INT', action='store', help='Number of folds in cross validation',
                        type=int, default=2)
groupClass.add_argument('--CL_nb_class', metavar='INT', action='store', help='Number of classes, -1 for all', type=int,
                        default=2)
groupClass.add_argument('--CL_classes', metavar='STRING', action='store', nargs="+",
                        help='Classes used in the dataset (names of the folders) if not filled, random classes will be '
                             'selected ex. walrus mole leopard', default=["yes","no"])
groupClass.add_argument('--CL_type', metavar='STRING', action='store',
                        help='Determine whether to use Multiview, Monoview, or Benchmark, separate with : if multiple',
                        default='Benchmark')
groupClass.add_argument('--CL_algos_monoview', metavar='STRING', action='store', nargs="+",
                        help='Determine which monoview classifier to use if empty, considering all',
                        default=[''])
groupClass.add_argument('--CL_algos_multiview', metavar='STRING', action='store', nargs="+",
                        help='Determine which multiview classifier to use if empty, considering all',
                        default=[''])
groupClass.add_argument('--CL_cores', metavar='INT', action='store', help='Number of cores, -1 for all', type=int,
                        default=2)
groupClass.add_argument('--CL_statsiter', metavar='INT', action='store',
                        help='Number of iteration for each algorithm to mean results', type=int,
                        default=2)
groupClass.add_argument('--CL_metrics', metavar='STRING', action='store', nargs="+",
                        help='Determine which metrics to use, separate metric and configuration with ":". If multiple, separate with space. If no metric is specified, considering all with accuracy for classification '
                             , default=[''])
groupClass.add_argument('--CL_metric_princ', metavar='STRING', action='store',
                        help='Determine which metric to use for randomSearch and optimization' , default="f1_score")
groupClass.add_argument('--CL_GS_iter', metavar='INT', action='store',
                        help='Determine how many Randomized grid search tests to do', type=int, default=2)
groupClass.add_argument('--CL_HPS_type', metavar='STRING', action='store',
                        help='Determine which hyperparamter search function use', default="randomizedSearch")

groupRF = parser.add_argument_group('Random Forest arguments')
groupRF.add_argument('--CL_RandomForest_trees', metavar='INT', type=int, action='store', help='Number max trees',
                     default=25)
groupRF.add_argument('--CL_RandomForest_max_depth', metavar='INT', type=int, action='store', help='Max depth for the trees',
                     default=5)
groupRF.add_argument('--CL_RandomForest_criterion', metavar='STRING', action='store', help='Criterion for the trees',
                     default="entropy")

groupSVMLinear = parser.add_argument_group('Linear SVM arguments')
groupSVMLinear.add_argument('--CL_SVMLinear_C', metavar='INT', type=int, action='store', help='Penalty parameter used',
                            default=1)

groupSVMRBF = parser.add_argument_group('SVW-RBF arguments')
groupSVMRBF.add_argument('--CL_SVMRBF_C', metavar='INT', type=int, action='store', help='Penalty parameter used',
                         default=1)

groupSVMPoly = parser.add_argument_group('Poly SVM arguments')
groupSVMPoly.add_argument('--CL_SVMPoly_C', metavar='INT', type=int, action='store', help='Penalty parameter used',
                          default=1)
groupSVMPoly.add_argument('--CL_SVMPoly_deg', metavar='INT', type=int, action='store', help='Degree parameter used',
                          default=2)

groupAdaboost = parser.add_argument_group('Adaboost arguments')
groupAdaboost.add_argument('--CL_Adaboost_n_est', metavar='INT', type=int, action='store', help='Number of estimators',
                           default=2)
groupAdaboost.add_argument('--CL_Adaboost_b_est', metavar='STRING', action='store', help='Estimators',
                           default='DecisionTreeClassifier')

groupDT = parser.add_argument_group('Decision Trees arguments')
groupDT.add_argument('--CL_DecisionTree_depth', metavar='INT', type=int, action='store',
                     help='Determine max depth for Decision Trees', default=3)
groupDT.add_argument('--CL_DecisionTree_criterion', metavar='STRING', action='store',
                     help='Determine max depth for Decision Trees', default="entropy")
groupDT.add_argument('--CL_DecisionTree_splitter', metavar='STRING', action='store',
                     help='Determine criterion for Decision Trees', default="random")


groupSGD = parser.add_argument_group('SGD arguments')
groupSGD.add_argument('--CL_SGD_alpha', metavar='FLOAT', type=float, action='store',
                      help='Determine alpha for SGDClassifier', default=0.1)
groupSGD.add_argument('--CL_SGD_loss', metavar='STRING', action='store',
                      help='Determine loss for SGDClassifier', default='log')
groupSGD.add_argument('--CL_SGD_penalty', metavar='STRING', action='store',
                      help='Determine penalty for SGDClassifier', default='l2')

groupKNN = parser.add_argument_group('KNN arguments')
groupKNN.add_argument('--CL_KNN_neigh', metavar='INT', type=int, action='store',
                      help='Determine number of neighbors for KNN', default=1)
groupKNN.add_argument('--CL_KNN_weights', metavar='STRING', action='store',
                      help='Determine number of neighbors for KNN', default="distance")
groupKNN.add_argument('--CL_KNN_algo', metavar='STRING', action='store',
                      help='Determine number of neighbors for KNN', default="auto")
groupKNN.add_argument('--CL_KNN_p', metavar='INT', type=int, action='store',
                      help='Determine number of neighbors for KNN', default=1)

groupSCM = parser.add_argument_group('SCM arguments')
groupSCM.add_argument('--CL_SCM_max_rules', metavar='INT', type=int, action='store',
                      help='Max number of rules for SCM', default=1)
groupSCM.add_argument('--CL_SCM_p', metavar='FLOAT', type=float, action='store',
                      help='Max number of rules for SCM', default=1.0)
groupSCM.add_argument('--CL_SCM_model_type', metavar='STRING', action='store',
                      help='Max number of rules for SCM', default="conjunction")


groupMumbo = parser.add_argument_group('Mumbo arguments')
groupMumbo.add_argument('--MU_types', metavar='STRING', action='store', nargs="+",
                        help='Determine which monoview classifier to use with Mumbo', default=['DecisionTree', 'DecisionTree', 'DecisionTree'])
groupMumbo.add_argument('--MU_config', metavar='STRING', action='store', nargs='+',
                        help='Configuration for the monoview classifier in Mumbo',
                        default=['2:0.5', '2:0.5', '2:0.5'])
groupMumbo.add_argument('--MU_iter', metavar='INT', action='store', nargs=3,
                        help='Max number of iteration, min number of iteration, convergence threshold', type=float,
                        default=[10, 1, 0.01])

groupFusion = parser.add_argument_group('Fusion arguments')
groupFusion.add_argument('--FU_types', metavar='STRING', action='store', nargs="+",
                         help='Determine which type of fusion to use',
                         default=[''])
groupEarlyFusion = parser.add_argument_group('Early Fusion arguments')
groupEarlyFusion.add_argument('--FU_early_methods', metavar='STRING', action='store', nargs="+",
                         help='Determine which early fusion method of fusion to use',
                         default=[''])
groupEarlyFusion.add_argument('--FU_E_method_configs', metavar='STRING', action='store', nargs='+',
                         help='Configuration for the early fusion methods separate method by space and values by :',
                              default=[''])
groupEarlyFusion.add_argument('--FU_E_cl_config', metavar='STRING', action='store', nargs='+',
                         help='Configuration for the monoview classifiers used separate classifier by space '
                              'and configs must be of form argument1_name:value,argument2_name:value',
                              default=[''])
groupEarlyFusion.add_argument('--FU_E_cl_names', metavar='STRING', action='store', nargs='+',
                         help='Name of the classifiers used for each early fusion method', default=[''])


groupLateFusion = parser.add_argument_group('Late Early Fusion arguments')
groupLateFusion.add_argument('--FU_late_methods', metavar='STRING', action='store', nargs="+",
                         help='Determine which late fusion method of fusion to use',
                         default=[''])
groupLateFusion.add_argument('--FU_L_method_config', metavar='STRING', action='store', nargs='+',
                         help='Configuration for the fusion method', default=[''])
groupLateFusion.add_argument('--FU_L_cl_config', metavar='STRING', action='store', nargs='+',
                         help='Configuration for the monoview classifiers used', default=[''])
groupLateFusion.add_argument('--FU_L_cl_names', metavar='STRING', action='store', nargs="+",
                         help='Names of the classifier used for late fusion', default=[''])
groupLateFusion.add_argument('--FU_L_select_monoview', metavar='STRING', action='store',
                         help='Determine which method to use to select the monoview classifiers', default="intersect")

args = parser.parse_args()
os.nice(args.nice)
nbCores = args.CL_cores
statsIter = args.CL_statsiter
start = time.time()

if args.name not in ["MultiOmic", "ModifiedMultiOmic", "Caltech", "Fake", "Plausible", "KMultiOmic"]:
    getDatabase = getattr(DB, "getClassicDB" + args.type[1:])
else:
    getDatabase = getattr(DB, "get" + args.name + "DB" + args.type[1:])

hyperParamSearch = args.CL_HPS_type

directory = initLogFile(args)

DATASET, LABELS_DICTIONARY = getDatabase(args.views, args.pathF, args.name, args.CL_nb_class,
                                         args.CL_classes)

datasetFiles = initMultipleDatasets(args, nbCores)

views, viewsIndices, allViews = initViews(DATASET, args)
if not views:
    raise ValueError, "Empty views list, modify selected views to match dataset " + args.views
NB_VIEW = len(views)

NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]

metrics = [metric.split(":") for metric in args.CL_metrics]
if metrics == [[""]]:
    metricsNames = [name for _, name, isPackage
                    in pkgutil.iter_modules(['Metrics']) if not isPackage and name != "log_loss"]
    metrics = [[metricName] for metricName in metricsNames]
    metrics = arangeMetrics(metrics, args.CL_metric_princ)
for metricIndex, metric in enumerate(metrics):
    if len(metric) == 1:
        metrics[metricIndex] = [metric[0], None]

logging.info("Start:\t Finding all available mono- & multiview algorithms")

benchmark = initBenchmark(args)

# fusionMethodConfig = [args.FU_method_config[0].split(":"), "b"]

initKWARGS = initKWARGS(args, benchmark)

dataBaseTime = time.time() - start

argumentDictionaries = {"Monoview": [], "Multiview": []}
argumentDictionaries = initMonoviewArguments(benchmark, argumentDictionaries, views, allViews, DATASET, NB_CLASS,
                                             initKWARGS)

bestClassifiers = []
bestClassifiersConfigs = []
resultsMonoview = []
labelsNames = LABELS_DICTIONARY.values()
if nbCores > 1:
    nbExperiments = len(argumentDictionaries["Monoview"])
    for stepIndex in range(int(math.ceil(float(nbExperiments) / nbCores))):
        resultsMonoview += (Parallel(n_jobs=nbCores)(
            delayed(ExecMonoview_multicore)(directory, args.name, labelsNames, args.CL_split, args.CL_nbFolds,
                                            coreIndex, args.type, args.pathF, statsIter, hyperParamSearch=hyperParamSearch,
                                            metrics=metrics, nIter=args.CL_GS_iter,
                                            **argumentDictionaries["Monoview"][coreIndex + stepIndex * nbCores])
            for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores))))
    scores = [[result[1][2][metrics[0][0]][1] for result in resultsMonoview if result[0] == viewIndex] for viewIndex in
              viewsIndices]
    classifiersNames = [[result[1][0] for result in resultsMonoview if result[0] == viewIndex] for viewIndex in
                        viewsIndices]
    classifiersConfigs = [[result[1][1][:-1] for result in resultsMonoview if result[0] == viewIndex] for viewIndex in
                          viewsIndices]

else:
    resultsMonoview += ([ExecMonoview(directory, DATASET.get("View" + str(arguments["viewIndex"])),
                                      DATASET.get("Labels").value, args.name, labelsNames,
                                      args.CL_split, args.CL_nbFolds, 1, args.type, args.pathF, statsIter,
                                      hyperParamSearch=hyperParamSearch, metrics=metrics, nIter=args.CL_GS_iter,
                                      **arguments)
                         for arguments in argumentDictionaries["Monoview"]])
    scores = [[result[1][2][metrics[0][0]][1] for result in resultsMonoview if result[0] == viewIndex] for viewIndex
              in viewsIndices]
    classifiersNames = [[result[1][0] for result in resultsMonoview if result[0] == viewIndex] for viewIndex in
                        viewsIndices]
    classifiersConfigs = [[result[1][1][:-1] for result in resultsMonoview if result[0] == viewIndex] for viewIndex in
                          viewsIndices]
monoviewTime = time.time() - dataBaseTime - start


argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, scores, classifiersConfigs,
                                              classifiersNames, NB_VIEW, metrics[0], argumentDictionaries)

if nbCores > 1:
    resultsMultiview = []
    nbExperiments = len(argumentDictionaries["Multiview"])
    for stepIndex in range(int(math.ceil(float(nbExperiments) / nbCores))):
        resultsMultiview += Parallel(n_jobs=nbCores)(
            delayed(ExecMultiview_multicore)(directory, coreIndex, args.name, args.CL_split, args.CL_nbFolds, args.type,
                                             args.pathF,
                                             LABELS_DICTIONARY, statsIter, hyperParamSearch=hyperParamSearch,
                                             metrics=metrics, nIter=args.CL_GS_iter,
                                             **argumentDictionaries["Multiview"][stepIndex * nbCores + coreIndex])
            for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores)))
else:
    resultsMultiview = [
        ExecMultiview(directory, DATASET, args.name, args.CL_split, args.CL_nbFolds, 1, args.type, args.pathF,
                      LABELS_DICTIONARY, statsIter, hyperParamSearch=hyperParamSearch,
                      metrics=metrics, nIter=args.CL_GS_iter, **arguments) for arguments in
        argumentDictionaries["Multiview"]]
multiviewTime = time.time() - monoviewTime - dataBaseTime - start
if nbCores > 1:
    logging.debug("Start:\t Deleting " + str(nbCores) + " temporary datasets for multiprocessing")
    datasetFiles = DB.deleteHDF5(args.pathF, args.name, nbCores)
    logging.debug("Start:\t Deleting datasets for multiprocessing")
labels = np.array(
    [resultMonoview[1][3] for resultMonoview in resultsMonoview] + [resultMultiview[3] for resultMultiview in
                                                                    resultsMultiview]).transpose()
trueLabels = DATASET.get("Labels").value
times = [dataBaseTime, monoviewTime, multiviewTime]
# times=[]
results = (resultsMonoview, resultsMultiview)
analyzeLabels(labels, trueLabels, results, directory)
logging.debug("Start:\t Analyze Global Results")
resultAnalysis(benchmark, results, args.name, times, metrics, directory)
logging.debug("Done:\t Analyze Global Results")

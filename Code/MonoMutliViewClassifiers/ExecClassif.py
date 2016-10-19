# Import built-in modules
import argparse
import pkgutil                       # for TimeStamp in CSVFile
import os
import time                             # for time calculations
import operator
import itertools

# Import 3rd party modules
from joblib import Parallel, delayed
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import math
import time

# Import own modules
import Multiview
from Multiview.ExecMultiview import ExecMultiview, ExecMultiview_multicore
from Monoview.ExecClassifMonoView import ExecMonoview, ExecMonoview_multicore
import Multiview.GetMultiviewDb as DB
import Monoview
from ResultAnalysis import resultAnalysis
from Versions import testVersions
import MonoviewClassifiers

# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype



testVersions()

def initLogFile(args):
    directory = os.path.dirname(os.path.abspath(__file__)) + "/Results/"
    logFileName = time.strftime("%Y%m%d-%H%M%S") + "-CMultiV-" + args.CL_type + "-" + "_".join(args.views.split(":")) + "-" + args.name + \
                  "-LOG"
    logFile = directory + logFileName
    if os.path.isfile(logFile + ".log"):
        for i in range(1, 20):
            testFileName = logFileName + "-" + str(i) + ".log"
            if not (os.path.isfile(directory + testFileName)):
                logfile = directory + testFileName
                break
    else:
        logFile += ".log"
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=logFile, level=logging.DEBUG,
                        filemode='w')
    if args.log:
        logging.getLogger().addHandler(logging.StreamHandler())


def initMultipleDatasets(args, nbCores):
    if nbCores>1:
        if DB.datasetsAlreadyExist(args.pathF, args.name, nbCores):
            logging.debug("Info:\t Enough copies of the dataset are already available")
            pass
        else:
            logging.debug("Start:\t Creating "+str(nbCores)+" temporary datasets for multiprocessing")
            logging.warning(" WARNING : /!\ This may use a lot of HDD storage space : "+
                            str(os.path.getsize(args.pathF+args.name+".hdf5")*nbCores/float(1024)/1000/1000)+" Gbytes /!\ ")
            time.sleep(5)
            datasetFiles = DB.copyHDF5(args.pathF, args.name, nbCores)
            logging.debug("Start:\t Creating datasets for multiprocessing")
            return datasetFiles


def initViews(DATASET, args):
    NB_VIEW = DATASET.get("Metadata").attrs["nbView"]
    if args.views!="":
        allowedViews = args.views.split(":")
        allViews = [str(DATASET.get("View"+str(viewIndex)).attrs["name"]) for viewIndex in range(NB_VIEW)]
        views = [str(DATASET.get("View"+str(viewIndex)).attrs["name"]) for viewIndex in range(NB_VIEW) if str(DATASET.get("View"+str(viewIndex)).attrs["name"]) in allowedViews]
        viewsIndices = [viewIndex for viewIndex in range(NB_VIEW) if str(DATASET.get("View"+str(viewIndex)).attrs["name"]) in allowedViews]
        return views, viewsIndices, allViews
    else:
        views = [str(DATASET.get("View"+str(viewIndex)).attrs["name"]) for viewIndex in range(NB_VIEW)]
        viewsIndices = np.arange(NB_VIEW)
        allViews = views
        return views, viewsIndices, allViews


def initBenchmark(args):
    benchmark = {"Monoview":{}, "Multiview":[]}
    if args.CL_type.split(":")==["Benchmark"]:
        # if args.CL_algorithm=='':
        fusionModulesNames = [name for _, name, isPackage
                              in pkgutil.iter_modules(['Multiview/Fusion/Methods']) if not isPackage]
        fusionModules = [getattr(Multiview.Fusion.Methods, fusionModulesName)
                         for fusionModulesName in fusionModulesNames]
        fusionClasses = [getattr(fusionModule, fusionModulesName+"Classifier")
                         for fusionModulesName, fusionModule in zip(fusionModulesNames, fusionModules)]
        fusionMethods = dict((fusionModulesName, [name for _, name, isPackage in
                                                  pkgutil.iter_modules(["Multiview/Fusion/Methods/"+fusionModulesName+"Package"])
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
        if args.CL_algos_multiview.split(":") == [""]:
            algosMutliview = ["Mumbo", "Fusion"]
        else:
            algosMutliview = args.CL_algos_multiview.split(":")
        if "Mumbo" in algosMutliview:
            benchmark["Multiview"]["Mumbo"] = args.MU_types.split(":")
        if "Fusion" in algosMutliview:
            benchmark["Multiview"]["Fusion"]= {}
            benchmark["Multiview"]["Fusion"]["Methods"] = dict((fusionType, []) for fusionType in args.FU_types.split(":"))
            if "LateFusion" in args.FU_types.split(":"):
                if args.FU_late_methods.split(":") == [""]:
                    benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"] = [name for _, name, isPackage in
                                                                                 pkgutil.iter_modules(["Multiview/Fusion/Methods/LateFusionPackage"])
                                                                                 if not isPackage]
                else:
                    benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"] = args.FU_late_methods.split(":")
            if "EarlyFusion" in args.FU_types.split(":"):
                if args.FU_early_methods.split(":") == [""]:
                    benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] = [name for _, name, isPackage in
                                                                                  pkgutil.iter_modules(["Multiview/Fusion/Methods/EarlyFusionPackage"])
                                                                                  if not isPackage]
                else:
                    benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] = args.FU_early_methods.split(":")
            if args.CL_algos_monoview.split(":")==['']:
                benchmark["Multiview"]["Fusion"]["Classifiers"] = [name for _, name, isPackage in
                                                                   pkgutil.iter_modules(['MonoviewClassifiers'])
                                                                   if (not isPackage) and (name!="SGD") and (name[:3]!="SVM")
                                                                   and (name!="SCM")]
            else:
                benchmark["Multiview"]["Fusion"]["Classifiers"] = args.CL_algos_monoview.split(":")
    if "Monoview" in args.CL_type.strip(":"):
        benchmark["Monoview"] = args.CL_algos_monoview.split(":")
    return benchmark


def initMonoviewArguments(benchmark, argumentDictionaries, views, allViews, DATASET, NB_CLASS, kwargsInit):
    try:
        if benchmark["Monoview"]:
            argumentDictionaries["Monoview"] = []
            for view in views:
                for classifier in benchmark["Monoview"]:
                    if classifier=="SCM":
                        if DATASET.get("View"+str(allViews.index(view))).attrs["binary"]:
                            arguments = {"args":{classifier+"KWARGS": kwargsInit[classifier+"KWARGSInit"], "feat":view,
                                                 "CL_type": classifier, "nbClass":NB_CLASS}, "viewIndex":allViews.index(view)}
                            argumentDictionaries["Monoview"].append(arguments)
                        else:
                            pass
                    else:
                        arguments = {"args":{classifier+"KWARGS": kwargsInit[classifier+"KWARGSInit"], "feat":view,
                                             "CL_type": classifier, "nbClass":NB_CLASS}, "viewIndex":allViews.index(view)}
                        argumentDictionaries["Monoview"].append(arguments)
    except:
        pass
    return argumentDictionaries


def initKWARGS(args):
    kwargsInit={
        "RandomForestKWARGSInit" : {"0":map(int, args.CL_RF_trees.split())[0], "1":map(int, args.CL_RF_max_depth.split(":"))[0]},
        "SVMLinearKWARGSInit" : {"0":map(int, args.CL_SVML_C.split(":"))[0]},
        "SVMRBFKWARGSInit" : {"0":map(int, args.CL_SVMR_C.split(":"))[0]},
        "SVMPolyKWARGSInit" : {"0":map(int, args.CL_SVMP_C.split(":"))[0], '1':map(int, args.CL_SVMP_deg.split(":"))[0]},
        "DecisionTreeKWARGSInit" : {"0":map(int, args.CL_DT_depth.split(":"))[0]},
        "SGDKWARGSInit" : {"2": map(float, args.CL_SGD_alpha.split(":"))[0], "1": args.CL_SGD_penalty.split(":")[0],
                           "0":args.CL_SGD_loss.split(":")[0]},
        "KNNKWARGSInit" : {"0": map(float, args.CL_KNN_neigh.split(":"))[0]},
        "AdaboostKWARGSInit" : {"0": args.CL_Ada_n_est.split(":")[0], "1": args.CL_Ada_b_est.split(":")[0]},
        "SCMKWARGSInit" : {"0":args.CL_SCM_max_rules.split(":")[0]},
    }
    return kwargsInit


def initMultiviewArguments(args, benchmark, views, viewsIndices, accuracies, classifiersConfigs, classifiersNames, fusionMethodConfig, NB_VIEW):
    if benchmark["Multiview"]:
        try:
            if benchmark["Multiview"]["Fusion"]:
                if args.FU_cl_names.split(':') !=['']:
                    monoClassifiers = args.FU_cl_names.split(":")
                    monoClassifiersConfigs = [globals()[classifier+"KWARGS"] for classifier in monoClassifiers]
                    if args.FU_method_config != [""]:
                        fusionMethodConfigs = [map(float,config.split(":")) for config in args.FU_method_config]
                    elif not gridSearch:
                        raise ValueError("No config for fusion method given and no gridearch wanted")
                    else:
                        try:
                            fusionMethodConfigs = [["config"] for method in benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"]]
                        except:
                            pass
                    try:
                        for methodIndex, method in enumerate(benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"]):
                            if args.FU_fixed:
                                arguments = {"CL_type": "Fusion",
                                             "views": views,
                                             "NB_VIEW": len(views),
                                             "viewsIndices": viewsIndices,
                                             "NB_CLASS": len(args.CL_classes.split(":")),
                                             "LABELS_NAMES": args.CL_classes.split(":"),
                                             "FusionKWARGS": {"fusionType":"LateFusion", "fusionMethod":method,
                                                              "classifiersNames": args.FU_cl_names.split(":"),
                                                              "classifiersConfigs": monoClassifiersConfigs,
                                                              'fusionMethodConfig': fusionMethodConfigs[methodIndex], "nbView":(len(viewsIndices))}}
                                argumentDictionaries["Multiview"].append(arguments)
                            else:
                                for combination in itertools.combinations_with_replacement(range(len(monoClassifiers)), NB_VIEW):
                                    monoClassifiersNamesComb = [monoClassifiers[index] for index in combination]
                                    monoClassifiersConfigsComb = [monoClassifiersConfigs[index] for index in combination]
                                    arguments = {"CL_type": "Fusion",
                                                 "views": views,
                                                 "NB_VIEW": len(views),
                                                 "viewsIndices": viewsIndices,
                                                 "NB_CLASS": len(args.CL_classes.split(":")),
                                                 "LABELS_NAMES": args.CL_classes.split(":"),
                                                 "FusionKWARGS": {"fusionType":"LateFusion", "fusionMethod":method,
                                                                  "classifiersNames": monoClassifiersNamesComb,
                                                                  "classifiersConfigs": monoClassifiersConfigsComb,
                                                                  'fusionMethodConfig': fusionMethodConfigs[methodIndex], "nbView":(len(viewsIndices))}}
                                    argumentDictionaries["Multiview"].append(arguments)
                    except:
                        pass
                else:
                    try:
                        if benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"] and benchmark["Multiview"]["Fusion"]["Classifiers"]:
                            bestClassifiers = []
                            bestClassifiersConfigs = []
                            if argumentDictionaries["Monoview"] != []:
                                for viewIndex, view in enumerate(views):
                                    bestClassifiers.append(classifiersNames[viewIndex][np.argmax(np.array(accuracies[viewIndex]))])
                                    bestClassifiersConfigs.append(classifiersConfigs[viewIndex][np.argmax(np.array(accuracies[viewIndex]))])
                            for method in benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"]:
                                arguments = {"CL_type": "Fusion",
                                             "views": views,
                                             "NB_VIEW": len(views),
                                             "viewsIndices": viewsIndices,
                                             "NB_CLASS": len(args.CL_classes.split(":")),
                                             "LABELS_NAMES": args.CL_classes.split(":"),
                                             "FusionKWARGS": {"fusionType":"LateFusion", "fusionMethod":method,
                                                              "classifiersNames": bestClassifiers,
                                                              "classifiersConfigs": bestClassifiersConfigs,
                                                              'fusionMethodConfig': fusionMethodConfig, "nbView":(len(viewsIndices))}}
                                argumentDictionaries["Multiview"].append(arguments)
                    except:
                        pass
                try:
                    if benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] and benchmark["Multiview"]["Fusion"]["Classifiers"]:
                        for method in benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"]:
                            for classifier in benchmark["Multiview"]["Fusion"]["Classifiers"]:
                                arguments = {"CL_type": "Fusion",
                                             "views": views,
                                             "NB_VIEW": len(views),
                                             "viewsIndices": viewsIndices,
                                             "NB_CLASS": len(args.CL_classes.split(":")),
                                             "LABELS_NAMES": args.CL_classes.split(":"),
                                             "FusionKWARGS": {"fusionType":"EarlyFusion", "fusionMethod":method,
                                                              "classifiersNames": [classifier],
                                                              "classifiersConfigs": [initKWARGS[classifier+"KWARGSInit"]],
                                                              'fusionMethodConfig': fusionMethodConfig, "nbView":(len(viewsIndices))}}
                                argumentDictionaries["Multiview"].append(arguments)
                except:
                    pass
        except:
            pass
        try:
            if benchmark["Multiview"]["Mumbo"]:
                for combination in itertools.combinations_with_replacement(range(len(benchmark["Multiview"]["Mumbo"])), NB_VIEW):
                    mumboClassifiersNames = [benchmark["Multiview"]["Mumbo"][index] for index in combination]
                    arguments = {"CL_type": "Mumbo",
                                 "views": views,
                                 "NB_VIEW": len(views),
                                 "viewsIndices": viewsIndices,
                                 "NB_CLASS": len(args.CL_classes.split(":")),
                                 "LABELS_NAMES": args.CL_classes.split(":"),
                                 "MumboKWARGS": {"classifiersNames": mumboClassifiersNames,
                                                 "maxIter":int(args.MU_iter[0]), "minIter":int(args.MU_iter[1]),
                                                 "threshold":args.MU_iter[2],
                                                 "classifiersConfigs": [argument.split(":") for argument in args.MU_config], "nbView":(len(viewsIndices))}}
                    argumentDictionaries["Multiview"].append(arguments)
        except:
            pass
    return argumentDictionaries

parser = argparse.ArgumentParser(
    description='This file is used to benchmark the accuracies fo multiple classification algorithm on multiview data.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

groupStandard = parser.add_argument_group('Standard arguments')
groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
groupStandard.add_argument('--name', metavar='STRING', action='store', help='Name of Database (default: %(default)s)',
                           default='Plausible')
groupStandard.add_argument('--type', metavar='STRING', action='store', help='Type of database : .hdf5 or .csv (default: %(default)s)',
                           default='.hdf5')
groupStandard.add_argument('--views', metavar='STRING', action='store',help='Name of the views selected for learning (default: %(default)s)',
                           default='')
groupStandard.add_argument('--pathF', metavar='STRING', action='store',help='Path to the views (default: %(default)s)',
                           default='/home/bbauvin/Documents/Data/Data_multi_omics/')
groupStandard.add_argument('--nice', metavar='INT', action='store', type=int,
                           help='Niceness for the process', default=0)

groupClass = parser.add_argument_group('Classification arguments')
groupClass.add_argument('--CL_split', metavar='FLOAT', action='store',
                        help='Determine the learning rate if > 1.0, number of fold for cross validation', type=float,
                        default=0.7)
groupClass.add_argument('--CL_nbFolds', metavar='INT', action='store', help='Number of folds in cross validation',
                        type=int, default=2 )
groupClass.add_argument('--CL_nb_class', metavar='INT', action='store', help='Number of classes, -1 for all', type=int,
                        default=2)
groupClass.add_argument('--CL_classes', metavar='STRING', action='store',
                        help='Classes used in the dataset (names of the folders) if not filled, random classes will be '
                             'selected ex. walrus:mole:leopard', default="jambon:poney")
groupClass.add_argument('--CL_type', metavar='STRING', action='store',
                        help='Determine whether to use Multiview, Monoview, or Benchmark, separate with : if multiple',
                        default='Benchmark')
# groupClass.add_argument('--CL_algorithm', metavar='STRING', action='store',
#                         help='Determine which classifier to use, if empty, considering all', default='')
groupClass.add_argument('--CL_algos_monoview', metavar='STRING', action='store',
                        help='Determine which monoview classifier to use, separate with : if multiple, if empty, considering all', default='')
groupClass.add_argument('--CL_algos_multiview', metavar='STRING', action='store',
                        help='Determine which multiview classifier to use, separate with : if multiple, if empty, considering all', default='')
groupClass.add_argument('--CL_cores', metavar='INT', action='store', help='Number of cores, -1 for all', type=int,
                        default=1)
groupClass.add_argument('--CL_statsiter', metavar='INT', action='store', help='Number of iteration for each algorithm to mean results', type=int,
                        default=2)
groupClass.add_argument('--CL_metrics', metavar='STRING', action='store', nargs="+",
                        help='Determine which metrics to use, separate metric and configuration with ":". If multiple, separate with space. If no metric is specified, considering all with accuracy for classification '
                             'first one will be used for classification', default=[''])
groupClass.add_argument('--CL_GS_iter', metavar='INT', action='store',
                        help='Determine how many Randomized grid search tests to do', type=int, default=2)
groupClass.add_argument('--CL_GS_type', metavar='STRING', action='store',
                        help='Determine which hyperparamter search function use', default="randomizedSearch")

groupRF = parser.add_argument_group('Random Forest arguments')
groupRF.add_argument('--CL_RF_trees', metavar='STRING', action='store', help='Number max trees',
                     default='25')
groupRF.add_argument('--CL_RF_max_depth', metavar='STRING', action='store', help='Max depth for the trees',
                     default='5')

groupSVMLinear = parser.add_argument_group('Linear SVM arguments')
groupSVMLinear.add_argument('--CL_SVML_C', metavar='STRING', action='store', help='Penalty parameter used',
                      default='1')

groupSVMRBF = parser.add_argument_group('SVW-RBF arguments')
groupSVMRBF.add_argument('--CL_SVMR_C', metavar='STRING', action='store', help='Penalty parameter used',
                            default='1')

groupSVMPoly = parser.add_argument_group('Poly SVM arguments')
groupSVMPoly.add_argument('--CL_SVMP_C', metavar='STRING', action='store', help='Penalty parameter used',
                            default='1')
groupSVMPoly.add_argument('--CL_SVMP_deg', metavar='STRING', action='store', help='Degree parameter used',
                          default='2')

groupAdaboost = parser.add_argument_group('Adaboost arguments')
groupAdaboost.add_argument('--CL_Ada_n_est', metavar='STRING', action='store', help='Number of estimators',
                          default='2')
groupAdaboost.add_argument('--CL_Ada_b_est', metavar='STRING', action='store', help='Estimators',
                          default='DecisionTreeClassifier')

groupRF = parser.add_argument_group('Decision Trees arguments')
groupRF.add_argument('--CL_DT_depth', metavar='STRING', action='store',
                     help='Determine max depth for Decision Trees', default='3')

groupSGD = parser.add_argument_group('SGD arguments')
groupSGD.add_argument('--CL_SGD_alpha', metavar='STRING', action='store',
                      help='Determine alpha for SGDClassifier', default='0.1')
groupSGD.add_argument('--CL_SGD_loss', metavar='STRING', action='store',
                      help='Determine loss for SGDClassifier', default='log')
groupSGD.add_argument('--CL_SGD_penalty', metavar='STRING', action='store',
                      help='Determine penalty for SGDClassifier', default='l2')

groupSGD = parser.add_argument_group('KNN arguments')
groupSGD.add_argument('--CL_KNN_neigh', metavar='STRING', action='store',
                      help='Determine number of neighbors for KNN', default='1')

groupSGD = parser.add_argument_group('SCM arguments')
groupSGD.add_argument('--CL_SCM_max_rules', metavar='STRING', action='store',
                      help='Max number of rules for SCM', default='1')

groupMumbo = parser.add_argument_group('Mumbo arguments')
groupMumbo.add_argument('--MU_types', metavar='STRING', action='store',
                        help='Determine which monoview classifier to use with Mumbo',default='DecisionTree')
groupMumbo.add_argument('--MU_config', metavar='STRING', action='store', nargs='+',
                        help='Configuration for the monoview classifier in Mumbo',
                        default=[''])
groupMumbo.add_argument('--MU_iter', metavar='INT', action='store', nargs=3,
                        help='Max number of iteration, min number of iteration, convergence threshold', type=float,
                        default=[10,1, 0.01])

groupFusion = parser.add_argument_group('Fusion arguments')
groupFusion.add_argument('--FU_types', metavar='STRING', action='store',
                         help='Determine which type of fusion to use, if multiple separate with :',
                         default='LateFusion:EarlyFusion')
groupFusion.add_argument('--FU_early_methods', metavar='STRING', action='store',
                         help='Determine which early fusion method of fusion to use, if multiple separate with :',
                         default='')
groupFusion.add_argument('--FU_late_methods', metavar='STRING', action='store',
                         help='Determine which late fusion method of fusion to use, if multiple separate with :',
                         default='')
groupFusion.add_argument('--FU_method_config', metavar='STRING', action='store', nargs='+',
                         help='Configuration for the fusion method', default=[''])
groupFusion.add_argument('--FU_cl_config', metavar='STRING', action='store', nargs='+',
                         help='Configuration for the monoview classifiers used', default=[''])
groupFusion.add_argument('--FU_cl_names', metavar='STRING', action='store',
                         help='Names of the classifier used for fusion, one per view separated by :', default='')
groupFusion.add_argument('--FU_fixed', action='store_true',
                        help='Determine if you want fusion for the monoview classifier in the same order as written')


args = parser.parse_args()
os.nice(args.nice)
nbCores = args.CL_cores
statsIter = args.CL_statsiter
start = time.time()

if args.name not in ["MultiOmic", "ModifiedMultiOmic", "Caltech", "Fake", "Plausible", "KMultiOmic"]:
    getDatabase = getattr(DB, "getClassicDB" + args.type[1:])
else:
    getDatabase = getattr(DB, "get" + args.name + "DB" + args.type[1:])

try:
    gridSearch = args.CL_GS_type
except:
    gridSearch = False

initLogFile(args)

DATASET, LABELS_DICTIONARY = getDatabase(args.views.split(":"), args.pathF, args.name, len(args.CL_classes), args.CL_classes)

datasetFiles = initMultipleDatasets(args, nbCores)

views, viewsIndices, allViews = initViews(DATASET, args)
if not views:
    raise ValueError, "Empty views list, modify selected views to match dataset "+args.views
NB_VIEW = len(views)

NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]

metrics = [metric.split(":") for metric in args.CL_metrics]
if metrics == [[""]]:
    metricsNames = [name for _, name, isPackage
                    in pkgutil.iter_modules(['Metrics']) if not isPackage and name!="log_loss"]
    metrics = [[metricName] for metricName in metricsNames]
for metricIndex, metric in enumerate(metrics):
    if len(metric)==1:
        metrics[metricIndex]=[metric[0], None]


logging.info("Start:\t Finding all available mono- & multiview algorithms")

benchmark = initBenchmark(args)

fusionMethodConfig = [args.FU_method_config[0].split(":"), "b"]

initKWARGS = initKWARGS(args)

dataBaseTime = time.time()-start
argumentDictionaries = {"Monoview": {}, "Multiview": []}
argumentDictionaries = initMonoviewArguments(benchmark, argumentDictionaries, views, allViews, DATASET, NB_CLASS, initKWARGS)

bestClassifiers = []
bestClassifiersConfigs = []
resultsMonoview = []
labelsNames = LABELS_DICTIONARY.values()
if nbCores>1:
    nbExperiments = len(argumentDictionaries["Monoview"])
    for stepIndex in range(int(math.ceil(float(nbExperiments)/nbCores))):
        resultsMonoview+=(Parallel(n_jobs=nbCores)(
            delayed(ExecMonoview_multicore)(args.name, labelsNames, args.CL_split, args.CL_nbFolds, coreIndex, args.type, args.pathF, statsIter, gridSearch=gridSearch,
                                            metrics=metrics, nIter=args.CL_GS_iter, **argumentDictionaries["Monoview"][coreIndex + stepIndex * nbCores])
            for coreIndex in range(min(nbCores, nbExperiments - stepIndex  * nbCores))))
    accuracies = [[result[1][1] for result in resultsMonoview if result[0]==viewIndex] for viewIndex in range(NB_VIEW)]
    classifiersNames = [[result[1][0] for result in resultsMonoview if result[0]==viewIndex] for viewIndex in range(NB_VIEW)]
    classifiersConfigs = [[result[1][1][:-1] for result in resultsMonoview if result[0]==viewIndex] for viewIndex in range(NB_VIEW)]
else:
    resultsMonoview+=([ExecMonoview(DATASET.get("View"+str(arguments["viewIndex"])),
                                    DATASET.get("Labels").value, args.name, labelsNames,
                                    args.CL_split, args.CL_nbFolds, 1, args.type, args.pathF, statsIter,
                                    gridSearch=gridSearch, metrics=metrics, nIter=args.CL_GS_iter,
                                    **arguments)
                       for arguments in argumentDictionaries["Monoview"]])

    accuracies = [[result[1][2][metrics[0][0]][2] for result in resultsMonoview if result[0]==viewIndex] for viewIndex in viewsIndices]
    classifiersNames = [[result[1][0] for result in resultsMonoview if result[0]==viewIndex] for viewIndex in viewsIndices]
    classifiersConfigs = [[result[1][1][:-1] for result in resultsMonoview if result[0]==viewIndex] for viewIndex in viewsIndices]
monoviewTime = time.time()-dataBaseTime-start

argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, accuracies, classifiersConfigs, classifiersNames, fusionMethodConfig, NB_VIEW)

if nbCores>1:
    resultsMultiview = []
    nbExperiments = len(argumentDictionaries["Multiview"])
    for stepIndex in range(int(math.ceil(float(nbExperiments)/nbCores))):
        resultsMultiview += Parallel(n_jobs=nbCores)(
            delayed(ExecMultiview_multicore)(coreIndex, args.name, args.CL_split, args.CL_nbFolds, args.type, args.pathF,
                                   LABELS_DICTIONARY, statsIter, gridSearch=gridSearch,
                                   metrics=metrics, nIter=args.CL_GS_iter, **argumentDictionaries["Multiview"][stepIndex*nbCores+coreIndex])
            for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores)))
else:
    resultsMultiview = [ExecMultiview(DATASET, args.name, args.CL_split, args.CL_nbFolds, 1, args.type, args.pathF,
                               LABELS_DICTIONARY, statsIter, gridSearch=gridSearch,
                               metrics=metrics, nIter=args.CL_GS_iter, **arguments) for arguments in argumentDictionaries["Multiview"]]
multiviewTime = time.time()-monoviewTime-dataBaseTime-start
if nbCores>1:
    logging.debug("Start:\t Deleting "+str(nbCores)+" temporary datasets for multiprocessing")
    datasetFiles = DB.deleteHDF5(args.pathF, args.name, nbCores)
    logging.debug("Start:\t Deleting datasets for multiprocessing")

times = [dataBaseTime, monoviewTime, multiviewTime]
# times=[]
results = (resultsMonoview, resultsMultiview)
logging.debug("Start:\t Analyze Global Results")
resultAnalysis(benchmark, results, args.name, times, metrics)
logging.debug("Done:\t Analyze Global Results")


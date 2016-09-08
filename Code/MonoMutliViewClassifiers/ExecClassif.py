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

parser = argparse.ArgumentParser(
    description='This file is used to benchmark the accuracies fo multiple classification algorithm on multiview data.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

groupStandard = parser.add_argument_group('Standard arguments')
groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
groupStandard.add_argument('--name', metavar='STRING', action='store', help='Name of Database (default: %(default)s)',
                           default='MultiOmic')
groupStandard.add_argument('--type', metavar='STRING', action='store', help='Type of database : .hdf5 or .csv',
                           default='.hdf5')
groupStandard.add_argument('--views', metavar='STRING', action='store',help='Name of the views selected for learning',
                           default='Methyl:MiRNA_:RNASeq:Clinic')
groupStandard.add_argument('--pathF', metavar='STRING', action='store',help='Path to the views (default: %(default)s)',
                           default='/home/bbauvin/Documents/Data/Data_multi_omics/')
groupStandard.add_argument('--fileCL', metavar='STRING', action='store',
                           help='Name of classLabels CSV-file  (default: %(default)s)', default='classLabels.csv')
groupStandard.add_argument('--fileCLD', metavar='STRING', action='store',
                           help='Name of classLabels-Description CSV-file  (default: %(default)s)',
                           default='classLabels-Description.csv')
groupStandard.add_argument('--fileFeat', metavar='STRING', action='store',
                           help='Name of feature CSV-file  (default: %(default)s)', default='feature.csv')
groupStandard.add_argument('--nice', metavar='INT', action='store', type=int,
                           help='Niceness for the process', default=0)

groupClass = parser.add_argument_group('Classification arguments')
groupClass.add_argument('--CL_split', metavar='FLOAT', action='store',
                        help='Determine the learning rate if > 1.0, number of fold for cross validation', type=float,
                        default=0.7)
groupClass.add_argument('--CL_nbFolds', metavar='INT', action='store', help='Number of folds in cross validation',
                        type=int, default=5 )
groupClass.add_argument('--CL_nb_class', metavar='INT', action='store', help='Number of classes, -1 for all', type=int,
                        default=2)
groupClass.add_argument('--CL_classes', metavar='STRING', action='store',
                        help='Classes used in the dataset (names of the folders) if not filled, random classes will be '
                             'selected ex. walrus:mole:leopard', default="jambon:poney")
groupClass.add_argument('--CL_type', metavar='STRING', action='store',
                        help='Determine whether to use Multiview, Monoview, or Benchmark, separate with : if multiple',
                        default='Benchmark')
groupClass.add_argument('--CL_algorithm', metavar='STRING', action='store',
                        help='Determine which classifier to use, if empty, considering all', default='')
groupClass.add_argument('--CL_algos_monoview', metavar='STRING', action='store',
                        help='Determine which monoview classifier to use, separate with : if multiple, if empty, considering all', default='')
groupClass.add_argument('--CL_algos_multiview', metavar='STRING', action='store',
                        help='Determine which multiview classifier to use, separate with : if multiple, if empty, considering all', default='')
groupClass.add_argument('--CL_cores', metavar='INT', action='store', help='Number of cores, -1 for all', type=int,
                        default=1)
groupClass.add_argument('--CL_metrics', metavar='STRING', action='store', nargs="+",
                        help='Determine which metrics to use, separate metric and configuration with ":". If multiple, separate with space. If no metric is specified, considering all with accuracy for classification '
                             'first one will be used for classification', default=[''])
groupClass.add_argument('--CL_GS_iter', metavar='INT', action='store',
                        help='Determine how many Randomized grid search tests to do', type=int, default=30)
groupClass.add_argument('--CL_NoGS', action='store_false',
                        help='Determine how many Randomized grid search tests to do')

groupRF = parser.add_argument_group('Random Forest arguments')
groupRF.add_argument('--CL_RF_trees', metavar='STRING', action='store', help='GridSearch: Determine the trees',
                     default='25 75 125 175')
groupRF.add_argument('--CL_RF_max_depth', metavar='STRING', action='store', help='GridSearch: Determine the trees',
                     default='5:10:15')

groupSVMLinear = parser.add_argument_group('Linear SVM arguments')
groupSVMLinear.add_argument('--CL_SVML_C', metavar='STRING', action='store', help='GridSearch : Penalty parameters used',
                      default='1:10:100:1000')

groupSVMRBF = parser.add_argument_group('SVW-RBF arguments')
groupSVMRBF.add_argument('--CL_SVMR_C', metavar='STRING', action='store', help='GridSearch : Penalty parameters used',
                            default='1:10:100:1000')

groupSVMPoly = parser.add_argument_group('Poly SVM arguments')
groupSVMPoly.add_argument('--CL_SVMP_C', metavar='STRING', action='store', help='GridSearch : Penalty parameters used',
                            default='1:10:100:1000')
groupSVMPoly.add_argument('--CL_SVMP_deg', metavar='STRING', action='store', help='GridSearch : Degree parameters used',
                          default='1:2:5:10')

groupAdaboost = parser.add_argument_group('Adaboost arguments')
groupAdaboost.add_argument('--CL_Ada_n_est', metavar='STRING', action='store', help='GridSearch : Penalty parameters used',
                          default='1:10:100:1000')
groupAdaboost.add_argument('--CL_Ada_b_est', metavar='STRING', action='store', help='GridSearch : Degree parameters used',
                          default='DecisionTreeClassifier')

groupRF = parser.add_argument_group('Decision Trees arguments')
groupRF.add_argument('--CL_DT_depth', metavar='STRING', action='store',
                     help='GridSearch: Determine max depth for Decision Trees', default='1:3:5:7')

groupSGD = parser.add_argument_group('SGD arguments')
groupSGD.add_argument('--CL_SGD_alpha', metavar='STRING', action='store',
                      help='GridSearch: Determine alpha for SGDClassifier', default='0.1:0.2:0.5:0.9')
groupSGD.add_argument('--CL_SGD_loss', metavar='STRING', action='store',
                      help='GridSearch: Determine loss for SGDClassifier', default='log')
groupSGD.add_argument('--CL_SGD_penalty', metavar='STRING', action='store',
                      help='GridSearch: Determine penalty for SGDClassifier', default='l2')

groupSGD = parser.add_argument_group('KNN arguments')
groupSGD.add_argument('--CL_KNN_neigh', metavar='STRING', action='store',
                      help='GridSearch: Determine number of neighbors for KNN', default='1:5:10:15')

groupMumbo = parser.add_argument_group('Mumbo arguments')
groupMumbo.add_argument('--MU_types', metavar='STRING', action='store',
                        help='Determine which monoview classifier to use with Mumbo',default='DecisionTree')
groupMumbo.add_argument('--MU_config', metavar='STRING', action='store', nargs='+',
                        help='Configuration for the monoview classifier in Mumbo',
                        default=['3:1.0', '3:1.0', '3:1.0','3:1.0'])
groupMumbo.add_argument('--MU_iter', metavar='INT', action='store', nargs=3,
                        help='Max number of iteration, min number of iteration, convergence threshold', type=float,
                        default=[10,1, 0.01])

groupFusion = parser.add_argument_group('Fusion arguments')
groupFusion.add_argument('--FU_types', metavar='STRING', action='store',
                         help='Determine which type of fusion to use, if multiple separate with :',
                         default='LateFusion')
groupFusion.add_argument('--FU_ealy_methods', metavar='STRING', action='store',
                         help='Determine which early fusion method of fusion to use, if multiple separate with :',
                         default='WeightedLinear')
groupFusion.add_argument('--FU_late_methods', metavar='STRING', action='store',
                         help='Determine which late fusion method of fusion to use, if multiple separate with :',
                         default='WeightedLinear')
groupFusion.add_argument('--FU_method_config', metavar='STRING', action='store', nargs='+',
                         help='Configuration for the fusion method', default=['1:1:1:1'])
groupFusion.add_argument('--FU_cl_names', metavar='STRING', action='store',
                         help='Names of the monoview classifiers used',default='RandomForest:SGD:SVC:DecisionTree')
groupFusion.add_argument('--FU_cl_config', metavar='STRING', action='store', nargs='+',
                         help='Configuration for the monoview classifiers used', default=['3:4', 'log:l2', '10:linear','4'])


args = parser.parse_args()
os.nice(args.nice)
nbCores = args.CL_cores
start = time.time()
if args.name not in ["MultiOmic", "ModifiedMultiOmic", "Caltech", "Fake"]:
    getDatabase = getattr(DB, "getClassicDB" + args.type[1:])
else:
    getDatabase = getattr(DB, "get" + args.name + "DB" + args.type[1:])

try:
    gridSearch = args.CL_NoGS
except:
    gridSearch = True

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


DATASET, LABELS_DICTIONARY = getDatabase(args.views.split(":"), args.pathF, args.name, len(args.CL_classes), args.CL_classes)
datasetLength = DATASET.get("Metadata").attrs["datasetLength"]
if nbCores>1:
    logging.debug("Start:\t Creating "+str(nbCores)+" temporary datasets for multiprocessing")
    logging.warning(" WARNING : /!\ This may use a lot of HDD storage space : "+
                    str(os.path.getsize(args.pathF+args.name+".hdf5")*nbCores/float(1024)/1000/1000)+" Gbytes /!\ ")
    time.sleep(5)
    datasetFiles = DB.copyHDF5(args.pathF, args.name, nbCores)
    logging.debug("Start:\t Creating datasets for multiprocessing")

NB_VIEW = DATASET.get("Metadata").attrs["nbView"]
views = [str(DATASET.get("View"+str(viewIndex)).attrs["name"]) for viewIndex in range(NB_VIEW)]
NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]
metrics = [metric.split(":") for metric in args.CL_metrics]
if metrics == [[""]]:
    metricsNames = [name for _, name, isPackage
                    in pkgutil.iter_modules(['Metrics']) if not isPackage]
    metrics = [[metricName, None] for metricName in metricsNames]
print metrics

logging.info("Start:\t Finding all available mono- & multiview algorithms")
benchmark = {"Monoview":{}, "Multiview":[]}
if args.CL_type.split(":")==["Benchmark"]:
    if args.CL_algorithm=='':
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
                            if not isPackage]
        fusionMonoviewClassifiers = allMonoviewAlgos
        allFusionAlgos = {"Methods": fusionMethods, "Classifiers": fusionMonoviewClassifiers}
        allMumboAlgos = [name for _, name, isPackage in
                         pkgutil.iter_modules(['Multiview/Mumbo/Classifiers'])
                         if not isPackage and not name in ["SubSampling", "ModifiedMulticlass", "Kover"]]
        allMultiviewAlgos = {"Fusion": allFusionAlgos, "Mumbo": allMumboAlgos}
        benchmark = {"Monoview": allMonoviewAlgos, "Multiview": allMultiviewAlgos}

if "Multiview" in args.CL_type.strip(":"):
    benchmark["Multiview"] = {}
    if "Mumbo" in args.CL_algos_multiview.split(":"):
        benchmark["Multiview"]["Mumbo"] = args.MU_types.split(":")
    if "Fusion" in args.CL_algos_multiview.split(":"):
        benchmark["Multiview"]["Fusion"]= {}
        benchmark["Multiview"]["Fusion"]["Methods"] = dict((fusionType, []) for fusionType in args.FU_types.split(":"))
        if "LateFusion" in args.FU_types.split(":"):
            benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"] = args.FU_late_methods.split(":")
        if "EarlyFusion" in args.FU_types.split(":"):
            benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] = args.FU_early_methods.split(":")
        benchmark["Multiview"]["Fusion"]["Classifiers"] = args.FU_cl_names.split(":")


if "Monoview" in args.CL_type.strip(":"):
    benchmark["Monoview"] = args.CL_algos_monoview.split(":")


fusionClassifierConfig = "a"
fusionMethodConfig = [["0.25", "0.25", "0.25", "0.25"], "b"]
mumboClassifierConfig = "a"
mumboclassifierNames = "a"

RandomForestKWARGSInit = {"0":map(int, args.CL_RF_trees.split())[0], "1":map(int, args.CL_RF_max_depth.split(":"))[0]}
SVMLinearKWARGSInit = {"0":map(int, args.CL_SVML_C.split(":"))[0]}
SVMRBFKWARGSInit = {"0":map(int, args.CL_SVMR_C.split(":"))[0]}
SVMPolyKWARGSInit = {"0":map(int, args.CL_SVMP_C.split(":"))[0], '1':map(int, args.CL_SVMP_deg.split(":"))[0]}
DecisionTreeKWARGSInit = {"0":map(int, args.CL_DT_depth.split(":"))[0]}
SGDKWARGSInit = {"2": map(float, args.CL_SGD_alpha.split(":"))[0], "1": args.CL_SGD_penalty.split(":")[0],
             "0":args.CL_SGD_loss.split(":")[0]}
KNNKWARGSInit = {"0": map(float, args.CL_KNN_neigh.split(":"))[0]}
AdaboostKWARGSInit = {"0": args.CL_Ada_n_est.split(":")[0], "1": args.CL_Ada_b_est.split(":")[0]}

dataBaseTime = time.time()-start
argumentDictionaries = {"Monoview": {}, "Multiview": []}
try:
    if benchmark["Monoview"]:
        argumentDictionaries["Monoview"] = []
        for view in views:
            for classifier in benchmark["Monoview"]:
                arguments = {"args":{classifier+"KWARGS": globals()[classifier+"KWARGSInit"], "feat":view, "fileFeat": args.fileFeat,
                                     "fileCL": args.fileCL, "fileCLD": args.fileCLD, "CL_type": classifier}, "viewIndex":views.index(view)}
                argumentDictionaries["Monoview"].append(arguments)
except:
    pass
bestClassifiers = []
bestClassifiersConfigs = []
resultsMonoview = []
if nbCores>1:
    nbExperiments = len(argumentDictionaries["Monoview"])
    for stepIndex in range(int(math.ceil(float(nbExperiments)/nbCores))):
        resultsMonoview+=(Parallel(n_jobs=nbCores)(
                delayed(ExecMonoview_multicore)(args.name, args.CL_split, args.CL_nbFolds, coreIndex, args.type, args.pathF, gridSearch=gridSearch,
                                                metrics=metrics, nIter=args.CL_GS_iter, **argumentDictionaries["Monoview"][coreIndex + stepIndex * nbCores])
                for coreIndex in range(min(nbCores, nbExperiments - (stepIndex + 1) * nbCores))))
    accuracies = [[result[1][1] for result in resultsMonoview if result[0]==viewIndex] for viewIndex in range(NB_VIEW)]
    classifiersNames = [[result[1][0] for result in resultsMonoview if result[0]==viewIndex] for viewIndex in range(NB_VIEW)]
    classifiersConfigs = [[result[1][2] for result in resultsMonoview if result[0]==viewIndex] for viewIndex in range(NB_VIEW)]
    for viewIndex, view in enumerate(views):
        bestClassifiers.append(classifiersNames[viewIndex][np.argmax(np.array(accuracies[viewIndex]))])
        bestClassifiersConfigs.append(classifiersConfigs[viewIndex][np.argmax(np.array(accuracies[viewIndex]))])

else:
    resultsMonoview+=([ExecMonoview(DATASET.get("View"+str(arguments["viewIndex"])),
                                    DATASET.get("labels").value, args.name,
                                    args.CL_split, args.CL_nbFolds, 1, args.type, args.pathF,
                                    gridSearch=gridSearch, metrics=metrics, nIter=args.CL_GS_iter,
                                    **arguments)
                                for arguments in argumentDictionaries["Monoview"]])

    accuracies = [[result[1][1] for result in resultsMonoview if result[0]==viewIndex] for viewIndex in range(NB_VIEW)]
    classifiersNames = [[result[1][0] for result in resultsMonoview if result[0]==viewIndex] for viewIndex in range(NB_VIEW)]
    classifiersConfigs = [[result[1][2] for result in resultsMonoview if result[0]==viewIndex] for viewIndex in range(NB_VIEW)]
    for viewIndex, view in enumerate(views):
        bestClassifiers.append(classifiersNames[viewIndex][np.argmax(np.array(accuracies[viewIndex]))])
        bestClassifiersConfigs.append(classifiersConfigs[viewIndex][np.argmax(np.array(accuracies[viewIndex]))])
monoviewTime = time.time()-dataBaseTime
print resultsMonoview
try:
    if benchmark["Multiview"]:
        try:
            if benchmark["Multiview"]["Mumbo"]:
                for combination in itertools.combinations_with_replacement(range(len(benchmark["Multiview"]["Mumbo"])), NB_VIEW):
                    classifiersNames = [benchmark["Multiview"]["Mumbo"][index] for index in combination]
                    arguments = {"CL_type": "Mumbo",
                                 "views": args.views.split(":"),
                                 "NB_VIEW": len(args.views.split(":")),
                                 "NB_CLASS": len(args.CL_classes.split(":")),
                                 "LABELS_NAMES": args.CL_classes.split(":"),
                                 "MumboKWARGS": {"classifiersNames": classifiersNames,
                                                 "maxIter":int(args.MU_iter[0]), "minIter":int(args.MU_iter[1]),
                                                 "threshold":args.MU_iter[2],
                                                 "classifiersConfigs": [argument.split(":") for argument in args.MU_config]}}
                    argumentDictionaries["Multiview"].append(arguments)
        except:
            pass
        bestClassifiers = ["DecisionTree", "DecisionTree", "DecisionTree", "DecisionTree"]
        monoviewTime = 0
        resultsMonoview = []
        bestClassifiersConfigs = []
        try:
            if benchmark["Multiview"]["Fusion"]:
                try:
                    if benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"] and benchmark["Multiview"]["Fusion"]["Classifiers"]:
                        for method in benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"]:
                            arguments = {"CL_type": "Fusion",
                                         "views": args.views.split(":"),
                                         "NB_VIEW": len(args.views.split(":")),
                                         "NB_CLASS": len(args.CL_classes.split(":")),
                                         "LABELS_NAMES": args.CL_classes.split(":"),
                                         "FusionKWARGS": {"fusionType":"LateFusion", "fusionMethod":method,
                                                          "classifiersNames": bestClassifiers,
                                                          "classifiersConfigs": bestClassifiersConfigs,
                                                          'fusionMethodConfig': fusionMethodConfig}}
                            argumentDictionaries["Multiview"].append(arguments)
                except:
                    pass
                try:
                    if benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] and benchmark["Multiview"]["Fusion"]["Classifiers"]:
                        for method in benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"]:
                            for classifier in benchmark["Multiview"]["Fusion"]["Classifiers"]:
                                arguments = {"CL_type": "Fusion",
                                             "views": args.views.split(":"),
                                             "NB_VIEW": len(args.views.split(":")),
                                             "NB_CLASS": len(args.CL_classes.split(":")),
                                             "LABELS_NAMES": args.CL_classes.split(":"),
                                             "FusionKWARGS": {"fusionType":"EarlyFusion", "fusionMethod":method,
                                                              "classifiersNames": [classifier],
                                                              "classifiersConfigs": [globals()[classifier+"KWARGSInit"]],
                                                              'fusionMethodConfig': fusionMethodConfig}}
                                argumentDictionaries["Multiview"].append(arguments)
                except:
                    pass
        except:
            pass
except:
    pass
# resultsMultiview = []
if nbCores>1:
    resultsMultiview = []
    nbExperiments = len(argumentDictionaries["Multiview"])
    for stepIndex in range(int(math.ceil(float(nbExperiments)/nbCores))):
        resultsMultiview += Parallel(n_jobs=nbCores)(
            delayed(ExecMultiview_multicore)(coreIndex, args.name, args.CL_split, args.CL_nbFolds, args.type, args.pathF,
                                   LABELS_DICTIONARY, gridSearch=gridSearch,
                                   metrics=metrics, nIter=args.CL_GS_iter, **argumentDictionaries["Multiview"][stepIndex*nbCores+coreIndex])
            for coreIndex in range(min(nbCores, nbExperiments - (stepIndex + 1) * nbCores)))
else:
    resultsMultiview = [ExecMultiview(DATASET, args.name, args.CL_split, args.CL_nbFolds, 1, args.type, args.pathF,
                               LABELS_DICTIONARY, gridSearch=gridSearch,
                               metrics=metrics, nIter=args.CL_GS_iter, **arguments) for arguments in argumentDictionaries["Multiview"]]
multiviewTime = time.time()-monoviewTime
if nbCores>1:
    logging.debug("Start:\t Deleting "+str(nbCores)+" temporary datasets for multiprocessing")
    datasetFiles = DB.deleteHDF5(args.pathF, args.name, nbCores)
    logging.debug("Start:\t Deleting datasets for multiprocessing")

times = [dataBaseTime, monoviewTime, multiviewTime]
# times=[]
results = (resultsMonoview, resultsMultiview)
logging.debug("Start:\t Analyze Results")
resultAnalysis(benchmark, results, args.name, times, metrics)
logging.debug("Done:\t Analyze Results")


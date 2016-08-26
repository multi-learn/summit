import argparse
import pkgutil
import Multiview
from Multiview.ExecMultiview import ExecMultiview
from Monoview.ExecClassifMonoView import ExecMonoview
import Multiview.GetMultiviewDb as DB
import Monoview
import os
import time
import logging
from joblib import Parallel, delayed
from ResultAnalysis import resultAnalysis
import numpy as np
import MonoviewClassifiers


parser = argparse.ArgumentParser(
    description='This file is used to benchmark the accuracies fo multiple classification algorithm on multiview data.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

groupStandard = parser.add_argument_group('Standard arguments')
groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
groupStandard.add_argument('--name', metavar='STRING', action='store', help='Name of Database (default: %(default)s)',
                           default='ModifiedMultiOmic')
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

groupClass = parser.add_argument_group('Classification arguments')
groupClass.add_argument('--CL_split', metavar='FLOAT', action='store',
                        help='Determine the learning rate if > 1.0, number of fold for cross validation', type=float,
                        default=0.7)
groupClass.add_argument('--CL_nbFolds', metavar='INT', action='store', help='Number of folds in cross validation',
                        type=int, default=5 )
groupClass.add_argument('--CL_nb_class', metavar='INT', action='store', help='Number of classes, -1 for all', type=int,
                        default=4)
groupClass.add_argument('--CL_classes', metavar='STRING', action='store',
                        help='Classes used in the dataset (names of the folders) if not filled, random classes will be '
                             'selected ex. walrus:mole:leopard', default="")
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

groupRF = parser.add_argument_group('Random Forest arguments')
groupRF.add_argument('--CL_RF_trees', metavar='STRING', action='store', help='GridSearch: Determine the trees',
                     default='25 75 125 175')

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
                        default=[1000, 300, 0.0005])

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
nbCores = args.CL_cores

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

getDatabase = getattr(DB, "get" + args.name + "DB" + args.type[1:])
DATASET, LABELS_DICTIONARY = getDatabase(args.views, args.pathF, args.name, len(args.CL_classes), args.CL_classes)
datasetLength = DATASET.get("Metadata").attrs["datasetLength"]
NB_VIEW = DATASET.get("Metadata").attrs["nbView"]
views = [str(DATASET.get("View"+str(viewIndex)).attrs["name"]) for viewIndex in range(NB_VIEW)]
NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]


logging.info("Start:\t Finding all available mono- & multiview algorithms")
benchmark = {}
if args.CL_type.split(":")==["Benchmark"]:
    if args.CL_algorithm=='':
        fusionModulesNames = [name for _, name, isPackage
                              in pkgutil.iter_modules(['Multiview/Fusion/Methods']) if not isPackage]
        fusionModules = [getattr(Multiview.Fusion.Methods, fusionModulesName)
                         for fusionModulesName in fusionModulesNames]
        fusionClasses = [getattr(fusionModule, fusionModulesName+"Classifier")
                         for fusionModulesName, fusionModule in zip(fusionModulesNames, fusionModules)]
        fusionMethods = dict((fusionModulesName, [subclass.__name__ for subclass in fusionClasse.__subclasses__() ])
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
        benchmark = {"Monoview": allMonoviewAlgos, "Multiview" : allMultiviewAlgos}

if "Multiview" in args.CL_type.strip(":"):
    benchmark["Multiview"] = {}
    if "Mumbo" in args.CL_algos_multiview.split(":"):
        benchmark["Multiview"]["Mumbo"] = [args.MU_types.split(":")]
    if "Fusion" in args.CL_algo_multiview.split(":"):
        benchmark["Multiview"]["Fusion"]= {}
        benchmark["Multiview"]["Fusion"]["Methods"] = dict((fusionType, []) for fusionType in args.FU_types.split(":"))
        if "LateFusion" in args.FU_types.split(":"):
            benchmark["Multiview"]["Fusion"]["LateFusion"] = args.FU_late_methods.split(":")
        if "EarlyFusion" in args.FU_types.split(":"):
            benchmark["Multiview"]["Fusion"]["EarlyFusion"] = args.FU_early_methods.split(":")
        benchmark["Multiview"]["Fusion"]["Classifiers"] = args.FU_cl_names.split(":")


if "Monoview" in args.CL_type.strip(":"):
    benchmark["Monoview"] = args.CL_algos_monoview.split(":")


fusionClassifierConfig = "a"
fusionMethodConfig = "a"
mumboNB_ITER = 2
mumboClassifierConfig = "a"
mumboclassifierNames = "a"

RandomForestKWARGS = {"0":map(int, args.CL_RF_trees.split())}
SVMLinearKWARGS = {"0":map(int, args.CL_SVML_C.split(":"))}
SVMRBFKWARGS = {"0":map(int, args.CL_SVMR_C.split(":"))}
SVMPolyKWARGS = {"0":map(int, args.CL_SVMP_C.split(":")), '1':map(int, args.CL_SVMP_deg.split(":"))}
DecisionTreeKWARGS = {"0":map(int, args.CL_DT_depth.split(":"))}
SGDKWARGS = {"0": map(float, args.CL_SGD_alpha.split(":")), "1":args.CL_SGD_loss.split(":"),
             "2": args.CL_SGD_penalty.split(":")}
KNNKWARGS = {"0": map(float, args.CL_KNN_neigh.split(":"))}
AdaboostKWARGS = {"0": args.CL_Ada_n_est.split(":"), "1": args.CL_Ada_b_est.split(":")}


argumentDictionaries = {"Monoview": {}, "Multiview": []}
if benchmark["Monoview"]:
    for view in views:
        argumentDictionaries["Monoview"][str(view)] = []
        for classifier in benchmark["Monoview"]:

            arguments = {classifier+"KWARGS": globals()[classifier+"KWARGS"], "feat":view, "fileFeat": args.fileFeat,
                         "fileCL": args.fileCL, "fileCLD": args.fileCLD, "CL_type": classifier}

            argumentDictionaries["Monoview"][str(view)].append(arguments)
bestClassifiers = []
bestClassifiersConfigs = []
resultsMonoview =[]
for viewIndex, viewArguments in enumerate(argumentDictionaries["Monoview"].values()):
    resultsMonoview += (Parallel(n_jobs=nbCores)(
        delayed(ExecMonoview)(DATASET.get("View"+str(viewIndex)), DATASET.get("labels").value, args.name,
                              args.CL_split, args.CL_nbFolds, 1, args.type, args.pathF, gridSearch=True,
                              **arguments)
        for arguments in viewArguments))

    accuracies = [result[1] for result in resultsMonoview[viewIndex]]
    classifiersNames = [result[0] for result in resultsMonoview[viewIndex]]
    classifiersConfigs = [result[2] for result in resultsMonoview[viewIndex]]
    bestClassifiers.append(classifiersNames[np.argmax(np.array(accuracies))])
    bestClassifiersConfigs.append(classifiersConfigs[np.argmax(np.array(accuracies))])
# bestClassifiers = ["DecisionTree", "DecisionTree", "DecisionTree", "DecisionTree"]
# bestClassifiersConfigs = [["1"],["1"],["1"],["1"]]
#
# if benchmark["Multiview"]:
#     if benchmark["Multiview"]["Mumbo"]:
#         for classifier in benchmark["Multiview"]["Mumbo"]:
#             arguments = {"CL_type": "Mumbo",
#                          "views": args.views.split(":"),
#                          "NB_VIEW": len(args.views.split(":")),
#                          "NB_CLASS": len(args.CL_classes.split(":")),
#                          "LABELS_NAMES": args.CL_classes.split(":"),
#                          "MumboKWARGS": {"classifiersNames": ["DecisionTree", "DecisionTree", "DecisionTree",
#                                                               "DecisionTree"],
#                                          "maxIter":int(args.MU_iter[0]), "minIter":int(args.MU_iter[1]),
#                                          "threshold":args.MU_iter[2]}}
#             argumentDictionaries["Multiview"].append(arguments)
#     if benchmark["Multiview"]["Fusion"]:
#         if benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"] and benchmark["Multiview"]["Fusion"]["Classifiers"]:
#             for method in benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"]:
#                 arguments = {"CL_type": "Fusion",
#                              "views": args.views.split(":"),
#                              "NB_VIEW": len(args.views.split(":")),
#                              "NB_CLASS": len(args.CL_classes.split(":")),
#                              "LABELS_NAMES": args.CL_classes.split(":"),
#                              "FusionKWARGS": {"fusionType":"LateFusion", "fusionMethod":method,
#                                               "classifiersNames": bestClassifiers,
#                                               "classifiersConfigs": bestClassifiersConfigs,
#                                               'fusionMethodConfig': fusionMethodConfig}}
#                 argumentDictionaries["Multiview"].append(arguments)
#         if benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] and benchmark["Multiview"]["Fusion"]["Classifiers"]:
#             for method in benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"]:
#                 for classifier in benchmark["Multiview"]["Fusion"]["Classifiers"]:
#                     arguments = {"CL_type": "Fusion",
#                                  "views": args.views.split(":"),
#                                  "NB_VIEW": len(args.views.split(":")),
#                                  "NB_CLASS": len(args.CL_classes.split(":")),
#                                  "LABELS_NAMES": args.CL_classes.split(":"),
#                                  "FusionKWARGS": {"fusionType":"EarlyFusion", "fusionMethod":method,
#                                                   "classifiersNames": classifier,
#                                                   "classifiersConfigs": fusionClassifierConfig,
#                                                   'fusionMethodConfig': fusionMethodConfig}}
#                     argumentDictionaries["Multiview"].append(arguments)

# resultsMultiview = Parallel(n_jobs=nbCores)(
#     delayed(ExecMultiview)(DATASET, args.name, args.CL_split, args.CL_nbFolds, 1, args.type, args.pathF,
#                            LABELS_DICTIONARY, gridSearch=True, **arguments)
#     for arguments in argumentDictionaries["Multiview"])
resultsMultiview = []
results = (resultsMonoview, resultsMultiview)
resultAnalysis(benchmark, results)
print len(argumentDictionaries["Multiview"]), len(argumentDictionaries["Monoview"])



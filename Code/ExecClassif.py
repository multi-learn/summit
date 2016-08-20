import argparse
import pkgutil
import Multiview
from Multiview.ExecMultiview import ExecMultiview
from Monoview.ExecClassifMonoView import ExecMonoview
import Monoview
import os
import time
import logging
from joblib import Parallel, delayed
from ResultAnalysis import resultAnalysis
import numpy as np

parser = argparse.ArgumentParser(
    description='This file is used to benchmark the accuracies fo multiple classification algorithm on multiview data.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

groupStandard = parser.add_argument_group('Standard arguments')
groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
groupStandard.add_argument('--name', metavar='STRING', action='store', help='Name of Database (default: %(default)s)',
                           default='Caltech')
groupStandard.add_argument('--type', metavar='STRING', action='store', help='Type of database : .hdf5 or .csv',
                           default='.csv')
groupStandard.add_argument('--views', metavar='STRING', action='store',help='Name of the views selected for learning',
                           default='RGB:HOG:SIFT:HOG:MHOG')
groupStandard.add_argument('--pathF', metavar='STRING', action='store',help='Path to the views (default: %(default)s)',
                           default='../FeatExtraction/Results-FeatExtr/')
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
                        default=0.9)
groupClass.add_argument('--CL_nbFolds', metavar='INT', action='store', help='Number of folds in cross validation',
                        type=int, default=3)
groupClass.add_argument('--CL_nb_class', metavar='INT', action='store', help='Number of classes, -1 for all', type=int,
                        default=4)
groupClass.add_argument('--CL_classes', metavar='STRING', action='store',
                        help='Classes used in the dataset (names of the folders) if not filled, random classes will be '
                             'selected ex. walrus:mole:leopard', default="")
groupClass.add_argument('--CL_type', metavar='STRING', action='store',
                        help='Determine whether to use Multiview, Monoview, or Benchmark, separate with : if multiple',
                        default='Benchmark')
groupClass.add_argument('--CL_algorithm', metavar='STRING', action='store',
                        help='Determine which classifier to use, if CL_type = Benchmark, fill monoview and multiview '
                             'options', default='')
groupClass.add_argument('--CL_algos_monoview', metavar='STRING', action='store',
                        help='Determine which monoview classifier to use, separate with : if multiple', default='')
groupClass.add_argument('--CL_algos_multiview', metavar='STRING', action='store',
                        help='Determine which multiview classifier to use, separate with : if multiple', default='')
groupClass.add_argument('--CL_cores', metavar='INT', action='store', help='Number of cores, -1 for all', type=int,
                        default=5)

groupRF = parser.add_argument_group('Random Forest arguments')
groupRF.add_argument('--CL_RF_trees', metavar='STRING', action='store', help='GridSearch: Determine the trees',
                     default='25 75 125 175')

groupSVC = parser.add_argument_group('SVC arguments')
groupSVC.add_argument('--CL_SVC_kernel', metavar='STRING', action='store', help='GridSearch : Kernels used',
                      default='linear')
groupSVC.add_argument('--CL_SVC_C', metavar='STRING', action='store', help='GridSearch : Penalty parameters used',
                      default='1:10:100:1000')

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


groupMumbo = parser.add_argument_group('Mumbo arguments')
groupMumbo.add_argument('--MU_types', metavar='STRING', action='store',
                        help='Determine which monoview classifier to use with Mumbo',default='DecisionTree')
groupMumbo.add_argument('--MU_config', metavar='STRING', action='store', nargs='+',
                        help='Configuration for the monoview classifier in Mumbo',
                        default=['3:1.0', '3:1.0', '3:1.0','3:1.0'])
groupMumbo.add_argument('--MU_iter', metavar='INT', action='store',
                        help='Number of iterations in Mumbos learning process', type=int, default=5)

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
        fusionMonoviewClassifiers = [name for _, name, isPackage in
                                     pkgutil.iter_modules(['Multiview/Fusion/Methods/MonoviewClassifiers'])
                                     if not isPackage ]
        allFusionAlgos = {"Methods": fusionMethods, "Classifiers": fusionMonoviewClassifiers}
        allMumboAlgos = [name for _, name, isPackage in
                                   pkgutil.iter_modules(['Multiview/Mumbo/Classifiers'])
                                   if not isPackage and not name in ["SubSampling", "ModifiedMulticlass", "Kover"]]
        allMultiviewAlgos = {"Fusion": allFusionAlgos, "Mumbo": allMumboAlgos}
        allMonoviewAlgos = [key[15:] for key in dir(Monoview.ClassifMonoView) if key[:15] == "MonoviewClassif"]
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

classifierTable = "a"
fusionClassifierConfig = "a"
fusionMethodConfig = "a"
mumboNB_ITER = "a"
mumboClassifierConfig = "a"
mumboclassifierNames = "a"
RandomForestKWARGS = {"classifier__n_estimators":map(int, args.CL_RF_trees.split())}
SVCKWARGS = {"classifier__kernel":args.CL_SVC_kernel.split(":"), "classifier__C":map(int,args.CL_SVC_C.split(":"))}
DecisionTreeKWARGS = {"classifier__max_depth":map(int,args.CL_DT_depth.split(":"))}
SGDKWARGS = {"classifier__alpha" : map(float,args.CL_SGD_alpha.split(":")), "classifier__loss":args.CL_SGD_loss.split(":"),
             "classifier__penalty":args.CL_SGD_penalty.split(":")}


print benchmark
argumentDictionaries = {"Monoview":[], "Multiview":[]}
if benchmark["Monoview"]:
    for classifier in benchmark["Monoview"]:
        for view in args.views.split(":"):
            arguments = {classifier+"KWARGS": globals()[classifier+"KWARGS"], "feat":view, "fileFeat": args.fileFeat,
                         "fileCL": args.fileCL, "fileCLD": args.fileCLD, "CL_type": classifier}
            argumentDictionaries["Monoview"].append(arguments)
if benchmark["Multiview"]:
    if benchmark["Multiview"]["Fusion"]:
        if benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"] and benchmark["Multiview"]["Fusion"]["Classifiers"]:
            for method in benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"]:
                for i in range(int(np.power(len(args.views.split(":")), len(benchmark["Multiview"]["Fusion"]["Classifiers"])))):
                #for classifier in benchmark["Multiview"]["Fusion"]["Classifiers"]:
                  #  for view in args.views.split(":"):
                    if True==True:
                        classifiersMatrix = []
                        arguments = {"CL_type": "Fusion",
                                     "views": args.views.split(":"),
                                     "NB_VIEW": len(args.views.split(":")),
                                     "NB_CLASS": len(args.CL_classes.split(":")),
                                     "LABELS_NAMES": args.CL_classes.split(":"),
                                     "FusionKWARGS": {"fusionType":"LateFusion", "fusionMethod":method,
                                                      "monoviewClassifiersNames": classifierTable,
                                                      "monoviewClassifiersConfigs": fusionClassifierConfig,
                                                      'fusionMethodConfig': fusionMethodConfig}}
                        argumentDictionaries["Multiview"].append(arguments)
        if benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] and benchmark["Multiview"]["Fusion"]["Classifiers"]:
            for method in benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"]:
                for classifier in benchmark["Multiview"]["Fusion"]["Classifiers"]:
                    arguments = {"CL_type": "Fusion",
                                     "views": args.views.split(":"),
                                     "NB_VIEW": len(args.views.split(":")),
                                     "NB_CLASS": len(args.CL_classes.split(":")),
                                     "LABELS_NAMES": args.CL_classes.split(":"),
                                     "FusionKWARGS": {"fusionType":"EarlyFusion", "fusionMethod":method,
                                                      "monoviewClassifiersNames": classifier,
                                                      "monoviewClassifiersConfigs": fusionClassifierConfig,
                                                      'fusionMethodConfig': fusionMethodConfig}}
                    argumentDictionaries["Multiview"].append(arguments)
    if benchmark["Multiview"]["Mumbo"]:
        #for classifier in benchmark["Multiview"]["Mumbo"]:
        for i in range(int(np.power(len(args.views.split(":")), len(benchmark["Multiview"]["Mumbo"])))):
            arguments = {"CL_type": "Mumbo",
                         "views": args.views.split(":"),
                         "NB_VIEW": len(args.views.split(":")),
                         "NB_CLASS": len(args.CL_classes.split(":")),
                         "LABELS_NAMES": args.CL_classes.split(":"),
                         "MumboKWARGS": {"classifiersConfigs": mumboClassifierConfig,"NB_ITER": mumboNB_ITER,
                                         "classifiersNames": mumboclassifierNames}}
            argumentDictionaries["Multiview"].append(arguments)
results = {}
# for classifierType, argumentsList in argumentDictionaries.iteritems():
#     executionMethod = globals()["Exec"+classifierType]
#     results[classifierType] = Parallel(n_jobs=args.CL_cores)(delayed(executionMethod)
#                                                              (args.name, args.CL_split,args.CL_nbFolds, 1, args.type,
#                                                               args.pathF, **arguments)
#                                                              for arguments in argumentsList)
resultAnalysis(benchmark, results)
print len(argumentDictionaries["Multiview"]), len(argumentDictionaries["Monoview"])



# views = args.views.split(":")
# dataBaseType = args.type
# NB_VIEW = len(views)
# mumboClassifierConfig = [argument.split(':') for argument in args.MU_config]
#
# LEARNING_RATE = args.CL_split
# nbFolds = args.CL_nbFolds
# NB_CLASS = args.CL_nb_class
# LABELS_NAMES = args.CL_classes.split(":")
# mumboclassifierNames = args.MU_type.split(':')
# mumboNB_ITER = args.MU_iter
# NB_CORES = args.CL_cores
# fusionClassifierNames = args.FU_cl_names.split(":")
# fusionClassifierConfig = [argument.split(':') for argument in args.FU_cl_config]
# fusionMethodConfig = [argument.split(':') for argument in args.FU_method_config]
# FusionKWARGS = {"fusionType":args.FU_type, "fusionMethod":args.FU_method,
#                 "monoviewClassifiersNames":fusionClassifierNames, "monoviewClassifiersConfigs":fusionClassifierConfig,
#                 'fusionMethodConfig':fusionMethodConfig}
# MumboKWARGS = {"classifiersConfigs":mumboClassifierConfig, "NB_ITER":mumboNB_ITER, "classifiersNames":mumboclassifierNames}
# dir = os.path.dirname(os.path.abspath(__file__)) + "/Results/"
# logFileName = time.strftime("%Y%m%d-%H%M%S") + "-CMultiV-" + args.CL_type + "-" + "_".join(views) + "-" + args.name + \
#               "-LOG"
# logFile = dir + logFileName
# if os.path.isfile(logFile + ".log"):
#     for i in range(1, 20):
#         testFileName = logFileName + "-" + str(i) + ".log"
#         if not (os.path.isfile(dir + testFileName)):
#             logfile = dir + testFileName
#             break
# else:
#     logFile += ".log"
# logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=logFile, level=logging.DEBUG,
#                     filemode='w')
# if args.log:
#     logging.getLogger().addHandler(logging.StreamHandler())
#
# ExecMultiview(views, dataBaseType, args, NB_VIEW, LEARNING_RATE, nbFolds, NB_CLASS, LABELS_NAMES, NB_CORES,
#               MumboKWARGS, FusionKWARGS)
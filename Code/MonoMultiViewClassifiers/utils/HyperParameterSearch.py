import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools

from .. import Metrics


def searchBestSettings(dataset, labels, classifierPackage, classifierName, metrics, iLearningIndices, iKFolds, randomState, viewsIndices=None,
                       searchingTool="hyperParamSearch", nIter=1, **kwargs):
    """Used to select the right hyperparam optimization function to optimize hyper parameters"""
    if viewsIndices is None:
        viewsIndices = range(dataset.get("Metadata").attrs["nbView"])
    thismodule = sys.modules[__name__]
    searchingToolMethod = getattr(thismodule, searchingTool)
    bestSettings = searchingToolMethod(dataset, labels, classifierPackage, classifierName, metrics, iLearningIndices, iKFolds, randomState,
                                       viewsIndices=viewsIndices, nIter=nIter, **kwargs)
    return bestSettings  # or well set clasifier ?


def gridSearch(dataset, classifierName, viewsIndices=None, kFolds=None, nIter=1, **kwargs):
    """Used to perfom gridsearch on the classifiers"""
    pass


def randomizedSearch(dataset, labels, classifierPackage, classifierName, metrics, learningIndices, KFolds, randomState, viewsIndices=None, nIter=1,
                     nbCores=1, **classificationKWARGS):
    """Used to perform a random search on the classifiers to optimize hyper parameters"""
    if viewsIndices is None:
        viewsIndices = range(dataset.get("Metadata").attrs["nbView"])
    metric = metrics[0]
    metricModule = getattr(Metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    classifierModule = getattr(classifierPackage, classifierName+"Module")
    classifierClass = getattr(classifierModule, classifierName+"Class")
    if classifierName != "Mumbo":
        paramsSets = classifierModule.genParamsSets(classificationKWARGS, randomState, nIter=nIter)
        if metricModule.getConfig()[-14] == "h":
            baseScore = -1000.0
            isBetter = "higher"
        else:
            baseScore = 1000.0
            isBetter = "lower"
        bestSettings = None
        kFolds = KFolds.split(learningIndices, labels[learningIndices])
        for paramsSet in paramsSets:
            scores = []
            for trainIndices, testIndices in kFolds:
                classifier = classifierClass(randomState, NB_CORES=nbCores, **classificationKWARGS)
                classifier.setParams(paramsSet)
                classifier.fit_hdf5(dataset, labels, trainIndices=learningIndices[trainIndices], viewsIndices=viewsIndices)
                testLabels = classifier.predict_hdf5(dataset, usedIndices=learningIndices[testIndices],
                                                     viewsIndices=viewsIndices)
                testScore = metricModule.score(labels[learningIndices[testIndices]], testLabels)
                scores.append(testScore)
            crossValScore = np.mean(np.array(scores))

            if isBetter == "higher" and crossValScore > baseScore:
                baseScore = crossValScore
                bestSettings = paramsSet
            elif isBetter == "lower" and crossValScore < baseScore:
                baseScore = crossValScore
                bestSettings = paramsSet
        classifier = classifierClass(randomState, NB_CORES=nbCores, **classificationKWARGS)
        classifier.setParams(bestSettings)
    # TODO : This must be corrected
    else:
        bestConfigs, _ = classifierModule.gridSearch_hdf5(dataset, viewsIndices, classificationKWARGS, learningIndices,
                                                          randomState, metric=metric, nIter=nIter)
        classificationKWARGS["classifiersConfigs"] = bestConfigs
        classifier = classifierClass(randomState, NB_CORES=nbCores, **classificationKWARGS)

    return classifier


def spearMint(dataset, classifierName, viewsIndices=None, kFolds=None, nIter=1, **kwargs):
    """Used to perform spearmint on the classifiers to optimize hyper parameters,
    longer than randomsearch (can't be parallelized)"""
    pass


def genHeatMaps(params, scoresArray, outputFileName):
    """Used to generate a heat map for each doublet of hyperparms optimized on the previous function"""
    nbParams = len(params)
    if nbParams > 2:
        combinations = itertools.combinations(range(nbParams), 2)
    else:
        combinations = [(0, 1)]
    for combination in combinations:
        paramName1, paramArray1 = params[combination[0]]
        paramName2, paramArray2 = params[combination[1]]

        paramArray1Set = np.sort(np.array(list(set(paramArray1))))
        paramArray2Set = np.sort(np.array(list(set(paramArray2))))

        scoresMatrix = np.zeros((len(paramArray2Set), len(paramArray1Set))) - 0.1
        for param1, param2, score in zip(paramArray1, paramArray2, scoresArray):
            param1Index, = np.where(paramArray1Set == param1)
            param2Index, = np.where(paramArray2Set == param2)
            scoresMatrix[int(param2Index), int(param1Index)] = score

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scoresMatrix, interpolation='nearest', cmap=plt.cm.hot,
                   )
        plt.xlabel(paramName1)
        plt.ylabel(paramName2)
        plt.colorbar()
        plt.xticks(np.arange(len(paramArray1Set)), paramArray1Set)
        plt.yticks(np.arange(len(paramArray2Set)), paramArray2Set, rotation=45)
        plt.title('Validation metric')
        plt.savefig(outputFileName + "heat_map-" + paramName1 + "-" + paramName2 + ".png")
        plt.close()

# nohup python ~/dev/git/spearmint/spearmint/main.py . &

# import json
# import numpy as np
# import math
#
# from os import system
# from os.path import join
#
#
# def run_kover(dataset, split, model_type, p, max_rules, output_dir):
#     outdir = join(output_dir, "%s_%f" % (model_type, p))
#     kover_command = "kover learn " \
#                     "--dataset '%s' " \
#                     "--split %s " \
#                     "--model-type %s " \
#                     "--p %f " \
#                     "--max-rules %d " \
#                     "--max-equiv-rules 10000 " \
#                     "--hp-choice cv " \
#                     "--random-seed 0 " \
#                     "--output-dir '%s' " \
#                     "--n-cpu 1 " \
#                     "-v" % (dataset,
#                             split,
#                             model_type,
#                             p,
#                             max_rules,
#                             outdir)
#
#     system(kover_command)
#
#     return json.load(open(join(outdir, "results.json")))["cv"]["best_hp"]["score"]
#
#
# def main(job_id, params):
#     print params
#
#     max_rules = params["MAX_RULES"][0]
#
#     species = params["SPECIES"][0]
#     antibiotic = params["ANTIBIOTIC"][0]
#     split = params["SPLIT"][0]
#
#     model_type = params["model_type"][0]
#
#     # LS31
#     if species == "saureus":
#         dataset_path = "/home/droale01/droale01-ls31/projects/genome_scm/data/earle_2016/saureus/kover_datasets/%s.kover" % antibiotic
#     else:
#         dataset_path = "/home/droale01/droale01-ls31/projects/genome_scm/genome_scm_paper/data/%s/%s.kover" % (species, antibiotic)
#
#     output_path = "/home/droale01/droale01-ls31/projects/genome_scm/manifold_scm/spearmint/vanilla_scm/%s/%s" % (species, antibiotic)
#
#     # MacBook
#     #dataset_path = "/Volumes/Einstein 1/kover_phylo/datasets/%s/%s.kover" % (species, antibiotic)
#     #output_path = "/Volumes/Einstein 1/manifold_scm/version2/%s_spearmint" % antibiotic
#
#     return run_kover(dataset=dataset_path,
#                      split=split,
#                      model_type=model_type,
#                      p=params["p"][0],
#                      max_rules=max_rules,
#                      output_dir=output_path)
# killall mongod && sleep 1 && rm -r database/* && rm mongo.log*
# mongod --fork --logpath mongo.log --dbpath database
#
# {
#     "language"        : "PYTHON",
#     "experiment-name" : "vanilla_scm_cdiff_azithromycin",
#     "polling-time"    : 1,
#     "resources" : {
#         "my-machine" : {
#             "scheduler"         : "local",
#             "max-concurrent"    : 5,
#             "max-finished-jobs" : 100
#         }
#     },
#     "tasks": {
#         "resistance" : {
#             "type"       : "OBJECTIVE",
#             "likelihood" : "NOISELESS",
#             "main-file"  : "spearmint_wrapper",
#             "resources"  : ["my-machine"]
#         }
#     },
#     "variables": {
#
#         "MAX_RULES" : {
#             "type" : "ENUM",
#             "size" : 1,
#             "options": [10]
#         },
#
#
#         "SPECIES" : {
#             "type" : "ENUM",
#             "size" : 1,
#             "options": ["cdiff"]
#         },
#         "ANTIBIOTIC" : {
#             "type" : "ENUM",
#             "size" : 1,
#             "options": ["azithromycin"]
#         },
#         "SPLIT" : {
#             "type" : "ENUM",
#             "size" : 1,
#             "options": ["split_seed_2"]
#         },
#
#
#         "model_type" : {
#             "type" : "ENUM",
#             "size" : 1,
#             "options": ["conjunction", "disjunction"]
#         },
#         "p" : {
#             "type" : "FLOAT",
#             "size" : 1,
#             "min"  : 0.01,
#             "max"  : 100
#         }
#     }
# }

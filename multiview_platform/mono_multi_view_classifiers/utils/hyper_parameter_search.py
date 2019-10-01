import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import  randint
from sklearn.model_selection import RandomizedSearchCV


from .. import metrics


def searchBestSettings(dataset, labels, classifier_module, classifier_name,
                       metrics, learning_indices, iKFolds, random_state,
                       directory, viewsIndices=None, nb_cores=1,
                       searchingTool="randomized_search", n_iter=1,
                       classifier_config=None):
    """Used to select the right hyper-parameter optimization function
    to optimize hyper parameters"""
    if viewsIndices is None:
        viewsIndices = range(dataset.get("Metadata").attrs["nbView"])
    output_file_name = directory
    thismodule = sys.modules[__name__]
    if searchingTool is not "None":
        searchingToolMethod = getattr(thismodule, searchingTool)
        bestSettings, test_folds_preds = searchingToolMethod(dataset, labels, "multiview", random_state, output_file_name,
                                           classifier_module, classifier_name, iKFolds,
                                           nb_cores, metrics, n_iter, classifier_config,
                                           learning_indices=learning_indices, view_indices=viewsIndices,)
    else:
        bestSettings = classifier_config
    return bestSettings  # or well set clasifier ?


def gridSearch(dataset, classifierName, viewsIndices=None, kFolds=None, nIter=1,
               **kwargs):
    """Used to perfom gridsearch on the classifiers"""
    pass

class CustomRandint:
    """Used as a distribution returning a integer between low and high-1.
    It can be used with a multiplier agrument to be able to perform more complex generation
    for example 10 e -(randint)"""

    def __init__(self, low=0, high=0, multiplier=""):
        self.randint = randint(low, high)
        self.multiplier = multiplier

    def rvs(self, random_state=None):
        randinteger = self.randint.rvs(random_state=random_state)
        if self.multiplier == "e-":
            return 10 ** -randinteger
        else:
            return randinteger

    def get_nb_possibilities(self):
        return self.randint.b - self.randint.a

def compute_possible_combinations(params_dict):
    n_possibs = np.ones(len(params_dict)) * np.inf
    for value_index, value in enumerate(params_dict.values()):
        if type(value) == list:
            n_possibs[value_index] = len(value)
        elif isinstance(value, CustomRandint):
            n_possibs[value_index] = value.get_nb_possibilities()
    return n_possibs


def get_test_folds_preds(X, y, cv, estimator, framework, available_indices=None):
    test_folds_prediction = []
    if framework == "monoview":
        folds = cv.split(np.arange(len(y)), y)
    if framework == "multiview":
        folds = cv.split(available_indices, y[available_indices])
    fold_lengths = np.zeros(cv.n_splits, dtype=int)
    for fold_idx, (train_indices, test_indices) in enumerate(folds):
        fold_lengths[fold_idx] = len(test_indices)
        if framework == "monoview":
            estimator.fit(X[train_indices], y[train_indices])
            test_folds_prediction.append(estimator.predict(X[train_indices]))
        if framework == "multiview":
            estimator.fit(X, y, available_indices[train_indices])
            test_folds_prediction.append(
                estimator.predict(X, available_indices[test_indices]))
    minFoldLength = fold_lengths.min()
    test_folds_prediction = np.array(
        [test_fold_prediction[:minFoldLength] for test_fold_prediction in
         test_folds_prediction])
    return test_folds_prediction


def randomized_search(X, y, framework, random_state, output_file_name, classifier_module,
                         classifier_name, folds=4, nb_cores=1, metric=["accuracy_score", None], n_iter=30,
                         classifier_kwargs =None, learning_indices=None, view_indices=None):
    estimator = getattr(classifier_module, classifier_name)(random_state,
                                                           **classifier_kwargs)
    params_dict = estimator.genDistribs()
    if params_dict:
        metricModule = getattr(metrics, metric[0])
        if metric[1] is not None:
            metricKWARGS = dict((index, metricConfig) for index, metricConfig in
                                enumerate(metric[1]))
        else:
            metricKWARGS = {}
        scorer = metricModule.get_scorer(**metricKWARGS)
        nb_possible_combinations = compute_possible_combinations(params_dict)
        min_list = np.array(
            [min(nb_possible_combination, n_iter) for nb_possible_combination in
             nb_possible_combinations])
        random_search = MultiviewCompatibleRandomizedSearchCV(estimator,
                                                             n_iter=int(np.sum(min_list)),
                                                             param_distributions=params_dict,
                                                             refit=True,
                                                             n_jobs=nb_cores, scoring=scorer,
                                                             cv=folds, random_state=random_state,
                                                             learning_indices=learning_indices,
                                                             view_indices=view_indices,
                                                             framework = framework)
        random_search.fit(X, y)
        best_params = random_search.best_params_
        if "random_state" in best_params:
            best_params.pop("random_state")

        # bestParams = dict((key, value) for key, value in
        #                   estimator.genBestParams(detector).items() if
        #                   key is not "random_state")

        scoresArray = random_search.cv_results_['mean_test_score']
        params = [(key[6:], value ) for key, value in random_search.cv_results_.items() if key.startswith("param_")]
        # genHeatMaps(params, scoresArray, output_file_name)
        best_estimator = random_search.best_estimator_
    else:
        best_estimator = estimator
        best_params = {}
    testFoldsPreds = get_test_folds_preds(X, y, folds, best_estimator,
                                          framework, learning_indices)
    return best_params, testFoldsPreds


from sklearn.base import clone


class MultiviewCompatibleRandomizedSearchCV(RandomizedSearchCV):

    def __init__(self, estimator, param_distributions, n_iter=10,
                 refit=True, n_jobs=1, scoring=None, cv=None,
                 random_state=None, learning_indices=None, view_indices=None, framework="monoview"):
        super(MultiviewCompatibleRandomizedSearchCV, self).__init__(estimator,
                                                                    n_iter=n_iter,
                                                                    param_distributions=param_distributions,
                                                                    refit=refit,
                                                                    n_jobs=n_jobs, scoring=scoring,
                                                                    cv=cv, random_state=random_state)
        self.framework = framework
        self.available_indices = learning_indices
        self.view_indices = view_indices

    def fit(self, X, y=None, groups=None, **fit_params):
        if self.framework == "monoview":
            return super(MultiviewCompatibleRandomizedSearchCV, self).fit(X, y=y, groups=groups, **fit_params)
        elif self.framework == "multiview":
            return self.fit_multiview(X, y=y, groups=groups,**fit_params)

    def fit_multiview(self, X, y=None, groups=None, **fit_params):
        n_splits = self.cv.get_n_splits(self.available_indices, y[self.available_indices])
        folds = self.cv.split(self.available_indices, y[self.available_indices])
        candidate_params = list(self._get_param_iterator())
        base_estimator = clone(self.estimator)
        results = {}
        self.cv_results_ = dict(("param_"+param_name, []) for param_name in candidate_params[0].keys())
        self.cv_results_["mean_test_score"] = []
        for candidate_param_idx, candidate_param in enumerate(candidate_params):
            test_scores = np.zeros(n_splits)+1000
            for fold_idx, (train_indices, test_indices) in enumerate(folds):
                current_estimator = clone(base_estimator)
                current_estimator.set_params(**candidate_param)
                current_estimator.fit(X, y,
                                      train_indices=self.available_indices[train_indices],
                                      view_indices=self.view_indices)
                test_prediction = current_estimator.predict(
                    X,
                    self.available_indices[test_indices],
                    view_indices=self.view_indices)
                test_score = self.scoring._score_func(y[self.available_indices[test_indices]],
                                                      test_prediction)
                test_scores[fold_idx] = test_score
            for param_name, param in candidate_param.items():
                self.cv_results_["param_"+param_name].append(param)
            cross_validation_score = np.mean(test_scores)
            self.cv_results_["mean_test_score"].append(cross_validation_score)
            results[candidate_param_idx] = cross_validation_score
            if cross_validation_score <= min(results.values()):
                self.best_params_ = candidate_params[candidate_param_idx]
                self.best_score_ = cross_validation_score
        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y, **fit_params)
        self.n_splits_ = n_splits
        return self

    def get_test_folds_preds(self, X, y, estimator):
        test_folds_prediction = []
        if self.framework=="monoview":
            folds = self.cv.split(np.arange(len(y)), y)
        if self.framework=="multiview":
            folds = self.cv.split(self.available_indices, y)
        fold_lengths = np.zeros(self.cv.n_splits, dtype=int)
        for fold_idx, (train_indices, test_indices) in enumerate(folds):
            fold_lengths[fold_idx] = len(test_indices)
            if self.framework == "monoview":
                estimator.fit(X[train_indices], y[train_indices])
                test_folds_prediction.append(estimator.predict(X[train_indices]))
            if self.framework =="multiview":
                estimator.fit(X, y, self.available_indices[train_indices])
                test_folds_prediction.append(estimator.predict(X, self.available_indices[test_indices]))
        minFoldLength = fold_lengths.min()
        test_folds_prediction = np.array(
            [test_fold_prediction[:minFoldLength] for test_fold_prediction in test_folds_prediction])
        return test_folds_prediction





def randomizedSearch(dataset, labels, classifierPackage, classifierName,
                     metrics_list, learningIndices, KFolds, randomState,
                     viewsIndices=None, nIter=1,
                     nbCores=1, **classificationKWARGS):
    """Used to perform a random search on the classifiers to optimize hyper parameters"""
    if viewsIndices is None:
        viewsIndices = range(dataset.get("Metadata").attrs["nbView"])
    metric = metrics_list[0]
    metricModule = getattr(metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in
                            enumerate(metric[1]))
    else:
        metricKWARGS = {}
    classifierModule = getattr(classifierPackage, classifierName + "Module")
    classifierClass = getattr(classifierModule, classifierName + "Class")
    if classifierName != "Mumbo":
        paramsSets = classifierModule.genParamsSets(classificationKWARGS,
                                                    randomState, nIter=nIter)
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
                classifier = classifierClass(randomState, NB_CORES=nbCores,
                                             **classificationKWARGS)
                classifier.setParams(paramsSet)
                classifier.fit_hdf5(dataset, labels,
                                    trainIndices=learningIndices[trainIndices],
                                    viewsIndices=viewsIndices)
                testLabels = classifier.predict_hdf5(dataset, usedIndices=
                learningIndices[testIndices],
                                                     viewsIndices=viewsIndices)
                testScore = metricModule.score(
                    labels[learningIndices[testIndices]], testLabels)
                scores.append(testScore)
            crossValScore = np.mean(np.array(scores))

            if isBetter == "higher" and crossValScore > baseScore:
                baseScore = crossValScore
                bestSettings = paramsSet
            elif isBetter == "lower" and crossValScore < baseScore:
                baseScore = crossValScore
                bestSettings = paramsSet
        classifier = classifierClass(randomState, NB_CORES=nbCores,
                                     **classificationKWARGS)
        classifier.setParams(bestSettings)

    # TODO : This must be corrected
    else:
        bestConfigs, _ = classifierModule.gridSearch_hdf5(dataset, labels,
                                                          viewsIndices,
                                                          classificationKWARGS,
                                                          learningIndices,
                                                          randomState,
                                                          metric=metric,
                                                          nIter=nIter)
        classificationKWARGS["classifiersConfigs"] = bestConfigs
        classifier = classifierClass(randomState, NB_CORES=nbCores,
                                     **classificationKWARGS)

    return classifier


def spearMint(dataset, classifierName, viewsIndices=None, kFolds=None, nIter=1,
              **kwargs):
    """Used to perform spearmint on the classifiers to optimize hyper parameters,
    longer than randomsearch (can't be parallelized)"""
    pass


def genHeatMaps(params, scoresArray, outputFileName):
    """Used to generate a heat map for each doublet of hyperparms optimized on the previous function"""
    nbParams = len(params)
    if nbParams > 2:
        combinations = itertools.combinations(range(nbParams), 2)
    elif nbParams == 2:
        combinations = [(0, 1)]
    else:
        combinations = [()]
    for combination in combinations:
        if combination:
            paramName1, paramArray1 = params[combination[0]]
            paramName2, paramArray2 = params[combination[1]]
        else:
            paramName1, paramArray1 = params[0]
            paramName2, paramArray2 = ("Control", np.array([0]))

        paramArray1Set = np.sort(np.array(list(set(paramArray1))))
        paramArray2Set = np.sort(np.array(list(set(paramArray2))))

        scoresMatrix = np.zeros(
            (len(paramArray2Set), len(paramArray1Set))) - 0.1
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
        plt.savefig(
            outputFileName + "heat_map-" + paramName1 + "-" + paramName2 + ".png", transparent=True)
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

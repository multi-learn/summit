# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2025
# -----------------
#
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2025 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
#
#
# Description:
# -----------
#
# Supervised MultiModal Integration Tool's Readme
# This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.
#
# Version:
# -------
#
# * summit-multi-learn version = 0.0.2
#
# Licence:
# -------
#
# License: New BSD License : BSD-3-Clause
#
#
# ######### COPYRIGHT #########
#
#
#
#
import traceback
import yaml
from abc import abstractmethod

import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, \
    ParameterGrid, ParameterSampler
from sklearn.base import clone, BaseEstimator

from .multiclass import MultiClassWrapper
from .organization import secure_file_path
from .base import get_metric
import traceback
from abc import abstractmethod

import numpy as np
import yaml
from scipy.stats import randint, uniform
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, \
    ParameterGrid, ParameterSampler

from .base import get_metric
from .multiclass import MultiClassWrapper
from .organization import secure_file_path


class HPSearch:

    def translate_param_distribs(self, param_distribs):
        translated_params = {}
        if param_distribs is None:
            return translated_params
        for param_name, value in param_distribs.items():
            if type(value) == list:
                translated_params[param_name] = value
            elif type(value)==dict:
                if "Uniform" in value.keys():
                    distrib = self.translate_uniform(value["Uniform"])
                elif "Randint" in value.keys():
                    distrib = self.translate_randint(value["Randint"])
                else:
                    distrib=value
                translated_params[param_name] = distrib
            else:
                translated_params[param_name] = value
        return translated_params

    def get_scoring(self, metric):
        if isinstance(metric, dict):
            metric_module, metric_kwargs = get_metric(metric)
            return metric_module.get_scorer(**metric_kwargs)
        else:
            return metric

    def fit_multiview(self, X, y, groups=None, **fit_params):
        print(self.available_indices)
        n_splits = self.cv.get_n_splits(self.available_indices,
                                        y[self.available_indices])
        folds = list(
            self.cv.split(self.available_indices, y[self.available_indices]))
        self.get_candidate_params(X)
        estimator = clone(self.estimator)
        results = {}
        self.cv_results_ = dict(("param_" + param_name, []) for param_name in
                                self.candidate_params[0].keys())
        self.cv_results_["mean_test_score"] = []
        self.cv_results_["params"] = []
        n_failed = 0
        self.tracebacks_params = []
        for candidate_param_idx, candidate_param in enumerate(
                self.candidate_params):
            test_scores = np.zeros(n_splits) + 1000
            try:
                for fold_idx, (train_indices,
                               test_indices) in enumerate(folds):
                    current_estimator = clone(estimator)
                    current_estimator.set_params(**candidate_param)
                    current_estimator.fit(X, y,
                                          train_indices=self.available_indices[
                                              train_indices],
                                          view_indices=self.view_indices)
                    test_prediction = current_estimator.predict(
                        X,
                        self.available_indices[test_indices],
                        view_indices=self.view_indices)
                    test_score = self.scoring._score_func(
                        y[self.available_indices[test_indices]],
                        test_prediction,
                        **self.scoring._kwargs)
                    test_scores[fold_idx] = test_score
                self.cv_results_['params'].append(
                    current_estimator.get_params())
                cross_validation_score = np.mean(test_scores)
                self.cv_results_["mean_test_score"].append(
                    cross_validation_score)
                results[candidate_param_idx] = cross_validation_score
                if cross_validation_score >= max(results.values()):
                    self.best_params_ = self.candidate_params[
                        candidate_param_idx]
                    self.best_score_ = cross_validation_score
            except BaseException:
                if self.track_tracebacks:
                    n_failed += 1
                    self.tracebacks.append(traceback.format_exc())
                    self.tracebacks_params.append(candidate_param)
                else:
                    raise
        if n_failed == self.n_iter:
            raise ValueError(
                'No fits were performed. All HP combination returned errors \n\n' + '\n'.join(
                    self.tracebacks))
        self.cv_results_["mean_test_score"] = np.array(
            self.cv_results_["mean_test_score"])
        if self.refit:
            self.best_estimator_ = clone(estimator).set_params(
                **self.best_params_)
            self.best_estimator_.fit(X, y, **fit_params)
        self.n_splits_ = n_splits
        return self

    @abstractmethod
    def get_candidate_params(self, X):  # pragma: no cover
        raise NotImplementedError

    def get_best_params(self):
        best_params = self.best_params_
        if "random_state" in best_params:
            best_params.pop("random_state")
        return best_params

    def gen_report(self, output_file_name):
        scores_array = self.cv_results_['mean_test_score']
        sorted_indices = np.argsort(-scores_array)
        tested_params = [self.cv_results_["params"][score_index]
                         for score_index in sorted_indices]
        scores_array = scores_array[sorted_indices]
        output_string = ""
        for parameters, score in zip(tested_params, scores_array):
            formatted_params = format_params(parameters)
            output_string += "\n{}\n\t\t{}".format(yaml.dump(formatted_params),
                                                   score)
        if self.tracebacks:
            output_string += "Failed : \n\n\n"
            for traceback, params in zip(self.tracebacks,
                                         self.tracebacks_params):
                output_string += '{}\n\n{}\n'.format(params, traceback)
        secure_file_path(output_file_name + "hps_report.txt")
        with open(output_file_name + "hps_report.txt", "w") as output_file:
            output_file.write(output_string)


class Random(RandomizedSearchCV, HPSearch):

    def __init__(self, estimator, param_distributions=None, n_iter=10,
                 refit=False, n_jobs=1, scoring=None, cv=None,  available_indices=None,
                 random_state=None, view_indices=None,
                 framework="monoview",
                 equivalent_draws=True, track_tracebacks=True):
        param_distributions = self.get_param_distribs(estimator, param_distributions)


        scoring = HPSearch.get_scoring(self, scoring)
        RandomizedSearchCV.__init__(self, estimator, n_iter=n_iter,
                                    param_distributions=param_distributions,
                                    refit=refit, n_jobs=n_jobs, scoring=scoring,
                                    cv=cv, random_state=random_state)
        self.framework = framework
        self.available_indices = available_indices
        self.view_indices = view_indices
        self.equivalent_draws = equivalent_draws
        self.track_tracebacks = track_tracebacks
        self.tracebacks = []

    def translate_uniform(self, args):
        return CustomUniform(**args)

    def translate_randint(self, args):
        return CustomRandint(**args)


    def get_param_distribs(self, estimator, user_distribs):
        user_distribs = self.translate_param_distribs(user_distribs)
        if isinstance(estimator, MultiClassWrapper):
            base_distribs = estimator.estimator.gen_distribs()
        else:
            base_distribs = estimator.gen_distribs()
        for key, value in user_distribs.items():
            base_distribs[key] = value
        return base_distribs

    def fit(self, X, y=None, groups=None, **fit_params):  # pragma: no cover
        if self.framework == "monoview":
            return RandomizedSearchCV.fit(self, X, y=y, groups=groups,
                                          **fit_params)

        elif self.framework == "multiview":
            return HPSearch.fit_multiview(self, X, y=y, groups=groups,
                                          **fit_params)

    def get_candidate_params(self, X):
        if self.equivalent_draws:
            self.n_iter = self.n_iter * X.nb_view
        self.candidate_params = list(
            ParameterSampler(self.param_distributions, self.n_iter,
                             random_state=self.random_state))



class Grid(GridSearchCV, HPSearch):

    def __init__(self, estimator, param_grid={}, refit=False, n_jobs=1,
                 scoring=None, cv=None,
                 available_indices=None, view_indices=None, framework="monoview",
                 random_state=None, track_tracebacks=True):
        scoring = HPSearch.get_scoring(self, scoring)
        GridSearchCV.__init__(self, estimator, param_grid, scoring=scoring,
                              n_jobs=n_jobs, refit=refit,
                              cv=cv)
        self.framework = framework
        self.available_indices = available_indices
        self.view_indices = view_indices
        self.track_tracebacks = track_tracebacks
        self.tracebacks = []

    def fit(self, X, y=None, groups=None, **fit_params):
        if self.framework == "monoview":
            return GridSearchCV.fit(self, X, y=y, groups=groups,
                                    **fit_params)
        elif self.framework == "multiview":
            return HPSearch.fit_multiview(self, X, y=y, groups=groups,
                                          **fit_params)

    def get_candidate_params(self, X):
        self.candidate_params = list(ParameterGrid(self.param_grid))
        self.n_iter = len(self.candidate_params)

class CustomDist:

    def multiply(self, random_number):
        if self.multiplier == "e-":
            return 10 ** -random_number
        elif self.multiplier =="e":
            return 10**random_number
        elif type(self.multiplier) in [int, float]:
            return self.multiplier*random_number
        else:
            return random_number

class CustomRandint(CustomDist):
    """Used as a distribution returning a integer between low and high-1.
    It can be used with a multiplier agrument to be able to perform more complex generation
    for example 10 e -(randint)"""

    def __init__(self, low=0, high=0, multiplier=""):
        self.randint = randint(low, high)
        self.low = low
        self.high = high
        self.multiplier = multiplier

    def rvs(self, random_state=None):
        rand_integer = self.randint.rvs(random_state=random_state)
        return self.multiply(rand_integer)

    def get_nb_possibilities(self):
        return self.high - self.low


class CustomUniform(CustomDist):
    """Used as a distribution returning a float between loc and loc + scale..
        It can be used with a multiplier agrument to be able to perform more complex generation
        for example 10 e -(float)"""

    def __init__(self, loc=0, state=1, multiplier=""):
        self.uniform = uniform(loc, state)
        self.multiplier = multiplier

    def rvs(self, random_state=None):
        unif = self.uniform.rvs(random_state=random_state)
        return self.multiply(unif)




def format_params(params, pref=""):
    if isinstance(params, dict):
        dictionary = {}
        for key, value in params.items():
            if isinstance(value, np.random.RandomState):
                pass
            elif isinstance(value, BaseEstimator):
                dictionary[key] = value.__class__.__name__
                for second_key, second_value in format_params(
                        value.get_params()).items():
                    dictionary[str(key) + "__" + second_key] = second_value
            else:
                dictionary[str(key)] = format_params(value)
        return dictionary
    elif isinstance(params, np.ndarray):
        return [format_params(param) for param in params]
    elif isinstance(params, np.float64):
        return float(params)
    elif isinstance(params, np.int64):
        return int(params)
    elif isinstance(params, list):
        return [format_params(param) for param in params]
    elif isinstance(params, np.str_):
        return str(params)
    else:
        return params

# def randomized_search_(dataset_var, labels, classifier_package, classifier_name,
#                       metrics_list, learning_indices, k_folds, random_state,
#                       views_indices=None, n_iter=1,
#                       nb_cores=1, **classification_kargs):
#     """Used to perform a random search on the classifiers to optimize hyper parameters"""
#     if views_indices is None:
#         views_indices = range(dataset_var.get("Metadata").attrs["nbView"])
#     metric = metrics_list[0]
#     metric_module = getattr(metrics, metric[0])
#     if metric[1] is not None:
#         metric_kargs = dict((index, metricConfig) for index, metricConfig in
#                             enumerate(metric[1]))
#     else:
#         metric_kargs = {}
#     classifier_module = getattr(classifier_package, classifier_name + "Module")
#     classifier_class = getattr(classifier_module, classifier_name + "Class")
#     if classifier_name != "Mumbo":
#         params_sets = classifier_module.gen_params_sets(classification_kargs,
#                                                     random_state, n_iter=n_iter)
#         if metric_module.getConfig()[-14] == "h":
#             base_score = -1000.0
#             is_better = "higher"
#         else:
#             base_score = 1000.0
#             is_better = "lower"
#         best_settings = None
#         kk_folds = k_folds.split(learning_indices, labels[learning_indices])
#         for params_set in params_sets:
#             scores = []
#             for trainIndices, testIndices in kk_folds:
#                 classifier = classifier_class(random_state, nb_scores=nb_cores,
#                                              **classification_kargs)
#                 classifier.setParams(params_set)
#                 classifier.fit_hdf5(dataset_var, labels,
#                                     train_indices=learning_indices[trainIndices],
#                                     views_indices=views_indices)
#                 test_labels = classifier.predict_hdf5(dataset_var,
#                                                       used_indices=learning_indices[testIndices],
#                                                       views_indices=views_indices)
#                 test_score = metric_module.score(
#                     labels[learning_indices[testIndices]], test_labels)
#                 scores.append(test_score)
#             cross_val_score = np.mean(np.array(scores))
#
#             if is_better == "higher" and cross_val_score > base_score:
#                 base_score = cross_val_score
#                 best_settings = params_set
#             elif is_better == "lower" and cross_val_score < base_score:
#                 base_score = cross_val_score
#                 best_settings = params_set
#         classifier = classifier_class(random_state, nb_cores=nb_cores,
#                                      **classification_kargs)
#         classifier.setParams(best_settings)
#
#     # TODO : This must be corrected
#     else:
#         best_configs, _ = classifier_module.grid_search_hdf5(dataset_var, labels,
#                                                              views_indices,
#                                                              classification_kargs,
#                                                              learning_indices,
#                                                              random_state,
#                                                              metric=metric,
#                                                              nI_iter=n_iter)
#         classification_kargs["classifiersConfigs"] = best_configs
#         classifier = classifier_class(random_state, nb_cores=nb_cores,
#                                       **classification_kargs)
#
#     return classifier

#
# def compute_possible_combinations(params_dict):
#     n_possibs = np.ones(len(params_dict)) * np.inf
#     for value_index, value in enumerate(params_dict.values()):
#         if type(value) == list:
#             n_possibs[value_index] = len(value)
#         elif isinstance(value, CustomRandint):
#             n_possibs[value_index] = value.get_nb_possibilities()
#     return np.prod(n_possibs)


# def get_test_folds_preds(X, y, cv, estimator, framework,
#                          available_indices=None):
#     test_folds_prediction = []
#     if framework == "monoview":
#         folds = cv.split(np.arange(len(y)), y)
#     if framework == "multiview":
#         folds = cv.split(available_indices, y[available_indices])
#     fold_lengths = np.zeros(cv.n_splits, dtype=int)
#     for fold_idx, (train_indices, test_indices) in enumerate(folds):
#         fold_lengths[fold_idx] = len(test_indices)
#         if framework == "monoview":
#             estimator.fit(X[train_indices], y[train_indices])
#             test_folds_prediction.append(estimator.predict(X[train_indices]))
#         if framework == "multiview":
#             estimator.fit(X, y, available_indices[train_indices])
#             test_folds_prediction.append(
#                 estimator.predict(X, available_indices[test_indices]))
#     min_fold_length = fold_lengths.min()
#     test_folds_prediction = np.array(
#         [test_fold_prediction[:min_fold_length] for test_fold_prediction in
#          test_folds_prediction])
#     return test_folds_prediction


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

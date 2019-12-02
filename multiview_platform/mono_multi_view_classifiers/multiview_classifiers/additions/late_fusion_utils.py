import numpy as np
import warnings
from scipy.stats import uniform


from ...multiview.multiview_utils import BaseMultiviewClassifier, get_available_monoview_classifiers, get_monoview_classifier, get_examples_views_indices, ConfigGenerator
from .fusion_utils import BaseFusionClassifier


class ClassifierDistribution:

    def __init__(self, seed=42, available_classifiers=None):
        self.random_state = np.random.RandomState(seed)
        self.available_classifiers = available_classifiers

    def draw(self, nb_view):
        return self.random_state.choice(self.available_classifiers,
                                        size=nb_view, replace=True)


class ClassifierCombinator:

    def __init__(self, need_probas=False):
        self.available_classifiers = get_available_monoview_classifiers(need_probas)

    def rvs(self, random_state=None):
        return ClassifierDistribution(seed=random_state.randint(1),
                                      available_classifiers=self.available_classifiers)


class ConfigDistribution:

    def __init__(self, seed=42, available_classifiers=None):
        self.random_state = np.random.RandomState(seed)
        self.config_generator = ConfigGenerator(available_classifiers)

    def draw(self, nb_view):
        config_samples = [self.config_generator.rvs(self.random_state)
                          for _ in range(nb_view)]
        return config_samples


class MultipleConfigGenerator:

    def __init__(self,):
        self.available_classifiers = get_available_monoview_classifiers()

    def rvs(self, random_state=None):
        return ConfigDistribution(seed=random_state.randint(1),
                                  available_classifiers=self.available_classifiers)


class WeightDistribution:

    def __init__(self, seed=42, distribution_type="uniform"):
        self.random_state = np.random.RandomState(seed)
        self.distribution_type = distribution_type

    def draw(self, nb_view):
        if self.distribution_type=="uniform":
            return self.random_state.random_sample(nb_view)


class WeightsGenerator:

    def __init__(self, distibution_type="uniform"):
        self.distribution_type=distibution_type

    def rvs(self, random_state=None):
        return WeightDistribution(seed=random_state.randint(1),
                                  distribution_type=self.distribution_type)


class LateFusionClassifier(BaseMultiviewClassifier, BaseFusionClassifier):

    def __init__(self, random_state=None, classifier_names=None,
                 classifier_configs=None, nb_cores=1, weights=None):
        super(LateFusionClassifier, self).__init__(random_state)
        self.classifiers_names = classifier_names
        self.classifier_configs = classifier_configs
        self.nb_cores = nb_cores
        self.weights = weights
        self.param_names = ["classifier_names", "classifier_configs", "weights"]
        self.distribs =[ClassifierCombinator(need_probas=self.need_probas),
                        MultipleConfigGenerator(),
                        WeightsGenerator()]

    def fit(self, X, y, train_indices=None, view_indices=None):
        self.init_params(X.nb_view)

        train_indices, view_indices = get_examples_views_indices(X,
                                                                  train_indices,
                                                                  view_indices)
        self.monoview_estimators = [monoview_estimator.fit(X.get_v(view_index, train_indices),
                                                           y[train_indices])
                                    for view_index, monoview_estimator
                                    in zip(view_indices,
                                           self.monoview_estimators)]
        return self

    def init_params(self, nb_view):
        if self.weights is None:
            self.weights = np.ones(nb_view) / nb_view
        elif isinstance(self.weights, WeightDistribution):
            self.weights = self.weights.draw(nb_view)
        else:
            self.weights = self.weights/np.sum(self.weights)

        if isinstance(self.classifiers_names, ClassifierDistribution):
            self.classifiers_names = self.classifiers_names.draw(nb_view)
        elif self.classifiers_names is None:
            self.classifiers_names = ["decision_tree" for _ in range(nb_view)]

        if isinstance(self.classifier_configs, ConfigDistribution):
            self.classifier_configs = self.classifier_configs.draw(nb_view)
        elif isinstance(self.classifier_configs, dict):
            self.classifier_configs = [{classifier_name: self.classifier_configs[classifier_name]} for classifier_name in self.classifiers_names]

        self.monoview_estimators = [
            self.init_monoview_estimator(classifier_name,
                                         self.classifier_configs[classifier_index],
                                         classifier_index=classifier_index)
            for classifier_index, classifier_name
            in enumerate(self.classifiers_names)]

    # def verif_clf_views(self, classifier_names, nb_view):
    #     if classifier_names is None:
    #         if nb_view is None:
    #             raise AttributeError(self.__class__.__name__+" must have either classifier_names or nb_views provided.")
    #         else:
    #             self.classifiers_names = self.get_classifiers(get_available_monoview_classifiers(), nb_view)
    #     else:
    #         if nb_view is None:
    #             self.classifiers_names = classifier_names
    #         else:
    #             if len(classifier_names)==nb_view:
    #                 self.classifiers_names = classifier_names
    #             else:
    #                 warnings.warn("nb_view and classifier_names not matching, choosing nb_view random classifiers in classifier_names.", UserWarning)
    #                 self.classifiers_names = self.get_classifiers(classifier_names, nb_view)


    def get_classifiers(self, classifiers_names, nb_choices):
        return self.random_state.choice(classifiers_names, size=nb_choices, replace=True)

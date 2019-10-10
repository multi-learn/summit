import numpy as np
import warnings
from scipy.stats import uniform


from ...multiview.multiview_utils import BaseMultiviewClassifier, get_available_monoview_classifiers, get_monoview_classifier, get_examples_views_indices, ConfigGenerator
from .fusion_utils import BaseLateFusionClassifier


class ClassifierCombinator:

    def __init__(self, nb_view):
        self.nb_view = nb_view
        self.available_classifiers = get_available_monoview_classifiers()

    def rvs(self, random_state=None):
        classifier_names = random_state.choice(self.available_classifiers,
                                               size=self.nb_view, replace=True)
        return classifier_names


class MultipleConfigGenerator(ConfigGenerator):
    def __init__(self, nb_view):
        super(MultipleConfigGenerator, self).__init__(get_available_monoview_classifiers())
        self.nb_view = nb_view
        self.multiple_distribs = [self.distribs for _ in range(nb_view)]

    def rvs(self, random_state=None):
        config_samples = [super(MultipleConfigGenerator, self).rvs(random_state)
                          for _ in range(self.nb_view)]
        return config_samples

class WeightsGenerator:

    def __init__(self, nb_view):
        self.nb_view=nb_view
        self.uniform = uniform(loc=0, state=1)

    def rvs(self, random_state=None):
        return np.array([uniform.rvs(random_state=random_state)
                         for _ in range(self.nb_view)])


class LateFusionClassifier(BaseMultiviewClassifier, BaseLateFusionClassifier):

    def __init__(self, random_state=None, classifier_names=None,
                 classifier_configs=None, nb_cores=1, nb_view=None, weights=None):
        super(LateFusionClassifier, self).__init__(random_state)
        self.verif_clf_views(classifier_names, nb_view)
        print(classifier_names)
        self.nb_view = len(classifier_names)
        self.classifiers_names = classifier_names
        self.classifier_configs = classifier_configs
        self.monoview_estimators = [self.init_monoview_estimator(classifier_name, classifier_index)
                                    for classifier_index, classifier_name
                                    in enumerate(self.classifiers_names)]
        self.nb_cores = nb_cores
        self.accuracies = np.zeros(len(classifier_names))
        self.needProbas = False
        if weights is None:
            self.weights = np.ones(nb_view)/nb_view
        else:
            self.weights = weights
        self.param_names = ["classifier_names", "classifier_configs", "weights"]
        self.distribs =[ClassifierCombinator(self.nb_view),
                        MultipleConfigGenerator(self.nb_view),
                        WeightsGenerator(nb_view)]

    def fit(self, X, y, train_indices=None, views_indices=None):
        train_indices, views_indices = get_examples_views_indices(X,
                                                                  train_indices,
                                                                  views_indices)
        self.monoview_estimators = [monoview_estimator.fit(X.get_v(view_index, train_indices), y[train_indices]) for view_index, monoview_estimator in zip(views_indices, self.monoview_estimators)]
        return self

    def verif_clf_views(self, classifier_names, nb_view):
        if classifier_names is None:
            if nb_view is None:
                raise AttributeError(self.__class__.__name__+" must have either classifier_names or nb_views provided.")
            else:
                self.classifiers_names = self.get_classifiers(get_available_monoview_classifiers(), nb_view)
        else:
            if nb_view is None:
                self.classifiers_names = classifier_names
            else:
                if len(classifier_names)==nb_view:
                    self.classifiers_names = classifier_names
                else:
                    warnings.warn("nb_view and classifier_names not matching, choosing nb_view random classifiers in classifier_names.", UserWarning)
                    self.classifiers_names = self.get_classifiers(classifier_names, nb_view)


    def get_classifiers(self, classifiers_names, nb_choices):
        return self.random_state.choice(classifiers_names, size=nb_choices, replace=True)

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from .. import multiview_classifiers
from .. import monoview_classifiers



class MultiviewResult(object):
    def __init__(self, classifier_name, classifier_config,
                 metrics_scores, full_labels, test_labels_multiclass):
        self.classifier_name = classifier_name
        self.classifier_config = classifier_config
        self.metrics_scores = metrics_scores
        self.full_labels_pred = full_labels
        self.y_test_multiclass_pred = test_labels_multiclass

    def get_classifier_name(self):
        multiview_classifier_module = getattr(multiview_classifiers,
                                            self.classifier_name)
        multiview_classifier = getattr(multiview_classifier_module,
                                       multiview_classifier_module.classifier_class_name)(42)
        return multiview_classifier.short_name


def get_names(classed_list):
    return np.array([object_.__class__.__name__ for object_ in classed_list])


class BaseMultiviewClassifier(BaseEstimator, ClassifierMixin):
    """
    BaseMultiviewClassifier base of Multiview classifiers

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.
    """

    def __init__(self, random_state):

        self.random_state = random_state
        self.short_name = self.__class__.__name__
        self.weird_strings = {}

    def gen_best_params(self, detector):
        """
        return best parameters of detector
        Parameters
        ----------
        detector :

        Returns
        -------
        best param : dictionary with param name as key and best parameters
            value
        """
        return dict((param_name, detector.best_params_[param_name])
                    for param_name in self.param_names)

    def genParamsFromDetector(self, detector):
        if self.classed_params:
            classed_dict = dict((classed_param, get_names(
                detector.cv_results_["param_" + classed_param]))
                                for classed_param in self.classed_params)
        if self.param_names:
            return [(param_name,
                     np.array(detector.cv_results_["param_" + param_name]))
                    if param_name not in self.classed_params else (
                param_name, classed_dict[param_name])
                    for param_name in self.param_names]
        else:
            return [()]

    def genDistribs(self):
        return dict((param_name, distrib) for param_name, distrib in
                    zip(self.param_names, self.distribs))

    def params_to_string(self):
        return ", ".join(
                [param_name + " : " + self.to_str(param_name) for param_name in
                 self.param_names])

    def getConfig(self):
        if self.param_names:
            return "\n\t\t- " + self.__class__.__name__ + "with " + self.params_to_string()
        else:
            return "\n\t\t- " + self.__class__.__name__ + "with no config."

    def to_str(self, param_name):
        if param_name in self.weird_strings:
            string = ""
            if "class_name" in self.weird_strings[param_name] :
                string+=self.get_params()[param_name].__class__.__name__
            if "config" in self.weird_strings[param_name]:
                string += "( with "+ self.get_params()[param_name].params_to_string()+")"
            else:
                string+=self.weird_strings[param_name](
                    self.get_params()[param_name])
            return string
        else:
            return str(self.get_params()[param_name])

    def get_interpretation(self):
        return "No detailed interpretation function"


def get_examples_views_indices(dataset, examples_indices, view_indices, ):
    """This function  is used to get all the examples indices and view indices if needed"""
    if view_indices is None:
        view_indices = np.arange(dataset.nb_view)
    if examples_indices is None:
        examples_indices = range(dataset.get_nb_examples())
    return examples_indices, view_indices


class ConfigGenerator():

    def __init__(self, classifier_names):
        self.distribs = {}
        for classifier_name in classifier_names:
            classifier_class = get_monoview_classifier(classifier_name)
            self.distribs[classifier_name] = dict((param_name, param_distrib)
                                  for param_name, param_distrib in
                                  zip(classifier_class().param_names,
                                      classifier_class().distribs)
                                if param_name!="random_state")

    def rvs(self, random_state=None):
        config_sample = {}
        for classifier_name, classifier_config in self.distribs.items():
            config_sample[classifier_name] = {}
            for param_name, param_distrib in classifier_config.items():
                if hasattr(param_distrib, "rvs"):
                    config_sample[classifier_name][param_name]=param_distrib.rvs(random_state=random_state)
                else:
                    config_sample[classifier_name][
                        param_name] = param_distrib[random_state.randint(len(param_distrib))]
        return config_sample


def get_available_monoview_classifiers(need_probas=False):
    available_classifiers = [module_name
                         for module_name in dir(monoview_classifiers)
                         if not module_name.startswith("__")]
    if need_probas:
        proba_classifiers = []
        for module_name in available_classifiers:
            module = getattr(monoview_classifiers, module_name)
            classifier_class = getattr(module, module.classifier_class_name)()
            proba_prediction = getattr(classifier_class, "predict_proba", None)
            if callable(proba_prediction):
                proba_classifiers.append(module_name)
        available_classifiers = proba_classifiers
    return available_classifiers

def get_monoview_classifier(classifier_name):
    classifier_module = getattr(monoview_classifiers, classifier_name)
    classifier_class = getattr(classifier_module, classifier_module.classifier_class_name)
    return classifier_class

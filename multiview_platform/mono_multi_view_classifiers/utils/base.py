import numpy as np
import pickle
from sklearn.base import BaseEstimator
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt


class BaseClassifier(BaseEstimator, ):

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
        return dict(
            (param_name, detector.best_params_[param_name]) for param_name in
            self.param_names)

    def gen_params_from_detector(self, detector):
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

    def gen_distribs(self):
        return dict((param_name, distrib) for param_name, distrib in
                    zip(self.param_names, self.distribs))

    def params_to_string(self):
        return ", ".join(
                [param_name + " : " + self.to_str(param_name) for param_name in
                 self.param_names])

    def get_config(self):
        if self.param_names:
            return "\n\t\t- " + self.__class__.__name__ + "with " + self.params_to_string()
        else:
            return "\n\t\t- " + self.__class__.__name__ + "with no config."

    def to_str(self, param_name):
        if param_name in self.weird_strings:
            if self.weird_strings[param_name] == "class_name":
                return self.get_params()[param_name].__class__.__name__
            else:
                return self.weird_strings[param_name](
                    self.get_params()[param_name])
        else:
            return str(self.get_params()[param_name])

    def get_interpretation(self, directory, y_test, multi_class=False):
        return ""

    def accepts_multi_class(self, random_state, n_samples=10, dim=2,
                           n_classes=3):
        if int(n_samples / n_classes) < 1:
            raise ValueError(
                "n_samples ({}) / n_classe ({}) must be over 1".format(
                    n_samples,
                    n_classes))
        if hasattr(self, "accepts_mutli_class"):
            return self.accepts_multi_class
        else:
            fake_mc_X = random_state.random_integers(low=0, high=100,
                                                     size=(n_samples, dim))
            fake_mc_y = [class_index
                         for _ in range(int(n_samples / n_classes))
                         for class_index in range(n_classes)]
            fake_mc_y += [0 for _ in range(n_samples % n_classes)]
            fake_mc_y = np.asarray(fake_mc_y)
            try:
                self.fit(fake_mc_X, fake_mc_y)
                self.predict(fake_mc_X)
                return True
            except ValueError:
                return False


def get_names(classed_list):
    return np.array([object_.__class__.__name__ for object_ in classed_list])

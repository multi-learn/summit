import numpy as np
from sklearn.base import BaseEstimator
from abc import abstractmethod
from datetime import timedelta as hms

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from multiview_platform.mono_multi_view_classifiers import metrics


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
        """
        Formats the parameters of the classifier as a string
        """
        return ", ".join(
            [param_name + " : " + self.to_str(param_name) for param_name in
             self.param_names])

    def get_config(self):
        """
        Generates a string to containing all the information about the
        classifier's configuration
        """
        if self.param_names:
            return self.__class__.__name__ + " with " + self.params_to_string()
        else:
            return self.__class__.__name__ + "with no config."

    def get_base_estimator(self, base_estimator, estimator_config):
        if estimator_config is None:
            estimator_config = {}
        if base_estimator is None:
            return DecisionTreeClassifier(**estimator_config)
        if isinstance(base_estimator, str):
            if base_estimator == "DecisionTreeClassifier":
                return DecisionTreeClassifier(**estimator_config)
            elif base_estimator == "AdaboostClassifier":
                return AdaBoostClassifier(**estimator_config)
            elif base_estimator == "RandomForestClassifier":
                return RandomForestClassifier(**estimator_config)
            else:
                raise ValueError('Base estimator string {} does not match an available classifier.'.format(base_estimator))
        elif isinstance(base_estimator, BaseEstimator):
            return base_estimator.set_params(**estimator_config)
        else:
            raise ValueError('base_estimator must be either a string or a BaseEstimator child class, it is {}'.format(type(base_estimator)))


    def to_str(self, param_name):
        """
        Formats a parameter into a string
        """
        if param_name in self.weird_strings:
            if self.weird_strings[param_name] == "class_name":
                return self.get_params()[param_name].__class__.__name__
            else:
                return self.weird_strings[param_name](
                    self.get_params()[param_name])
        else:
            return str(self.get_params()[param_name])

    def get_interpretation(self, directory, y_test, multi_class=False):
        """
        Base method that returns an empty string if there is not interpretation
        method in the classifier's module
        """
        return ""

    def accepts_multi_class(self, random_state, n_samples=10, dim=2,
                            n_classes=3):
        """
        Base function to test if the classifier accepts a multiclass task.
        It is highly recommended to overwrite it with a simple method that
        returns True or False in the classifier's module, as it will speed up
        the benchmark
        """
        if int(n_samples / n_classes) < 1:
            raise ValueError(
                "n_samples ({}) / n_class ({}) must be over 1".format(
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


def get_metric(metric_list):
    """
    Fetches the metric module in the metrics package
    """
    metric_module = getattr(metrics, metric_list[0][0])
    if metric_list[0][1] is not None:
        metric_kwargs = dict((index, metricConfig) for index, metricConfig in
                             enumerate(metric_list[0][1]))
    else:
        metric_kwargs = {}
    return metric_module, metric_kwargs


class ResultAnalyser():
    """
    A shared result analysis tool for mono and multiview classifiers.
    The main utility of this class is to generate a txt file summarizing
    the results and possible interpretation for the classifier.
    """

    def __init__(self, classifier, classification_indices, k_folds,
                 hps_method, metrics_list, n_iter, class_label_names,
                 train_pred, test_pred, directory, labels, database_name,
                 nb_cores, duration):
        """

        Parameters
        ----------
        classifier: estimator used for classification

        classification_indices: list of indices for train test sets

        k_folds: the sklearn StratifiedkFolds object

        hps_method: string naming the hyper-parameter search method

        metrics_list: list of the metrics to compute on the results

        n_iter: number of HPS iterations

        class_label_names: list of the names of the labels

        train_pred: classifier's prediction on the training set

        test_pred: classifier's prediction on the testing set

        directory: directory where to save the result analysis

        labels: the full labels array (Y in sklearn)

        database_name: the name of the database

        nb_cores: number of cores/threads use for the classification

        duration: duration of the classification
        """
        self.classifier = classifier
        self.train_indices, self.test_indices = classification_indices
        self.k_folds = k_folds
        self.hps_method = hps_method
        self.metrics_list = metrics_list
        self.n_iter = n_iter
        self.class_label_names = class_label_names
        self.train_pred = train_pred
        self.test_pred = test_pred
        self.directory = directory
        self.labels = labels
        self.string_analysis = ""
        self.database_name = database_name
        self.nb_cores = nb_cores
        self.duration = duration
        self.metric_scores = {}

    def get_all_metrics_scores(self, ):
        """
        Get the scores for all the metrics in the list
        Returns
        -------
        """
        for metric, metric_args in self.metrics_list:
            self.metric_scores[metric] = self.get_metric_scores(metric,
                                                                    metric_args)

    def get_metric_scores(self, metric, metric_kwargs):
        """
        Get the train and test scores for a specific metric and its arguments

        Parameters
        ----------

        metric : name of the metric, must be implemented in metrics

        metric_kwargs : the dictionary containing the arguments for the metric.

        Returns
        -------
        train_score, test_score
        """
        metric_module = getattr(metrics, metric)
        train_score = metric_module.score(y_true=self.labels[self.train_indices],
                                          y_pred=self.train_pred,
                                          **metric_kwargs)
        test_score = metric_module.score(y_true=self.labels[self.test_indices],
                                         y_pred=self.test_pred,
                                         **metric_kwargs)
        return train_score, test_score

    def print_metric_score(self,):
        """
        Generates a string, formatting the metrics configuration and scores

        Parameters
        ----------
        metric_scores : dictionary of train_score, test_score for each metric

        metric_list : list of metrics

        Returns
        -------
        metric_score_string string formatting all metric results
        """
        metric_score_string = "\n\n"
        for metric, metric_kwargs in self.metrics_list:
            metric_module = getattr(metrics, metric)
            metric_score_string += "\tFor {} : ".format(metric_module.get_config(
                **metric_kwargs))
            metric_score_string += "\n\t\t- Score on train : {}".format(self.metric_scores[metric][0])
            metric_score_string += "\n\t\t- Score on test : {}".format(self.metric_scores[metric][1])
            metric_score_string += "\n\n"
        return metric_score_string

    @abstractmethod
    def get_view_specific_info(self):
        pass

    @abstractmethod
    def get_base_string(self):
        pass

    def get_db_config_string(self,):
        """
        Generates a string, formatting all the information on the database

        Parameters
        ----------

        Returns
        -------
        db_config_string string, formatting all the information on the database
        """
        learning_ratio = len(self.train_indices) / (
                len(self.train_indices) + len(self.test_indices))
        db_config_string = "Database configuration : \n"
        db_config_string += "\t- Database name : {}\n".format(self.database_name)
        db_config_string += self.get_view_specific_info()
        db_config_string += "\t- Learning Rate : {}\n".format(learning_ratio)
        db_config_string += "\t- Labels used : " + ", ".join(
            self.class_label_names) + "\n"
        db_config_string += "\t- Number of cross validation folds : {}\n\n".format(self.k_folds.n_splits)
        return db_config_string

    def get_classifier_config_string(self, ):
        """
        Formats the information about the classifier and its configuration

        Returns
        -------
        A string explaining the classifier's configuration
        """
        classifier_config_string = "Classifier configuration : \n"
        classifier_config_string += "\t- " + self.classifier.get_config()+ "\n"
        classifier_config_string += "\t- Executed on {} core(s) \n".format(
            self.nb_cores)

        if self.hps_method.startswith('randomized_search'):
            classifier_config_string += "\t- Got configuration using randomized search with {}  iterations \n" .format(self.n_iter)
        return classifier_config_string

    def analyze(self, ):
        """
        Main function used in the monoview and multiview classification scripts

        Returns
        -------
        string_analysis : a string that will be stored in the log and in a txt
        file
        image_analysis : a list of images to save
        metric_scores : a dictionary of {metric: (train_score, test_score)}
        used in later analysis.
        """
        string_analysis = self.get_base_string()
        string_analysis += self.get_db_config_string()
        string_analysis += self.get_classifier_config_string()
        self.get_all_metrics_scores()
        string_analysis += self.print_metric_score()
        string_analysis += "\n\n Classification took {}".format(hms(seconds=int(self.duration)))
        string_analysis += "\n\n Classifier Interpretation : \n"
        string_analysis += self.classifier.get_interpretation(
            self.directory,
            self.labels[self.test_indices])
        image_analysis = {}
        return string_analysis, image_analysis, self.metric_scores


base_boosting_estimators = [DecisionTreeClassifier(max_depth=1),
                            DecisionTreeClassifier(max_depth=2),
                            DecisionTreeClassifier(max_depth=3),
                            DecisionTreeClassifier(max_depth=4),
                            DecisionTreeClassifier(max_depth=5), ]
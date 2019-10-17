import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ..metrics import zero_one_loss
from ..monoview.additions.BoostUtils import StumpsClassifiersGenerator, \
    BaseBoost
from ..monoview.monoview_utils import CustomRandint, \
    BaseMonoviewClassifier, change_label_to_minus, change_label_to_zero

classifier_class_name = "AdaboostGraalpy"

class AdaBoostGP(BaseEstimator, ClassifierMixin, BaseBoost):
    """Scikit-Learn compatible AdaBoost classifier. Original code by Pascal Germain, adapted by Jean-Francis Roy.


    Parameters
    ----------

    n_iterations : int, optional
        The number of iterations of the algorithm. Defaults to 200.

    iterations_to_collect_as_hyperparameters : list
        Iteration numbers to collect while learning, that will be converted as hyperparameter values at evaluation time.
        Defaults to None.
    classifiers_generator : Transformer, optional
        A transformer to convert input samples in voters' outputs. Default: Decision stumps transformer, with 10 stumps
        per attributes.
    callback_function : function, optional
        A function to call at each iteration that is supplied learning information. Defaults to None.

    n_stumps : int ( default : 10)

    self_complemented : boolean (default : True

    Attributes
    ----------
    n_iterations : int, optional
        The number of iterations of the algorithm. Defaults to 200.
    iterations_to_collect_as_hyperparameters : list
        Iteration numbers to collect while learning, that will be converted as hyperparameter values at evaluation time.
        Defaults to None.
    classifiers_generator : Transformer, optional
        A transformer to convert input samples in voters' outputs. Default: Decision stumps transformer, with 10 stumps
        per attributes.
    callback_function : function, optional
        A function to call at each iteration that is supplied learning information. Defaults to None.

    """

    def __init__(self, n_iterations=200,
                 iterations_to_collect_as_hyperparameters=True,
                 classifiers_generator=None, callback_function=None,
                 n_stumps=10, self_complemented=True):

        self.n_iterations = n_iterations
        self.n_stumps = n_stumps
        self.iterations_to_collect_as_hyperparameters = iterations_to_collect_as_hyperparameters
        self.estimators_generator = classifiers_generator
        self.callback_function = callback_function
        self.self_complemented = self_complemented

    def fit(self, X, y):
        """Fits the algorithm on training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples, )
            The input labels.

        Returns
        -------
        self

        """
        y_neg = change_label_to_minus(y)

        if self.estimators_generator is None:
            self.estimators_generator = StumpsClassifiersGenerator(
                n_stumps_per_attribute=self.n_stumps,
                self_complemented=self.self_complemented)

        # Step 1: We fit the classifiers generator and get its classification matrix.
        self.estimators_generator.fit(X, y_neg)
        # hint: This is equivalent to construct a new X
        classification_matrix = self._binary_classification_matrix(X)

        n_samples, n_voters = classification_matrix.shape
        # logging.debug("n_voters = {}".format(n_voters))

        # Step 2: We initialize the weights on the samples and the weak classifiers.
        sample_weights = np.ones(n_samples) / n_samples
        alpha_weights = np.zeros(n_voters)
        self.losses = []

        # Step 3: We loop for each iteration.
        self.collected_weight_vectors_ = []
        for t in range(self.n_iterations):

            # Step 4: We find the classifier that maximizes the success, weighted by the sample weights.
            classifier_successes = np.dot(classification_matrix.T,
                                          sample_weights * y_neg)

            best_voter_index = np.argmax(classifier_successes)
            success = classifier_successes[best_voter_index]

            if success >= 1.0:
                logging.info("AdaBoost stopped : perfect classifier found!")
                self.weights_ = np.zeros(n_voters)
                self.weights_[best_voter_index] = 1.0
                return self

            # Step 5: We calculate the alpha_t parameter and update the alpha weights.
            alpha = 0.5 * np.log((1.0 + success) / (1.0 - success))
            alpha_weights[best_voter_index] += alpha

            # logging.debug("{} : {}".format(t, str(alpha)))

            # Step 6: We update the sample weights.
            sample_weights *= np.exp(
                -1 * alpha * y_neg * classification_matrix[:, best_voter_index])

            normalization_constant = sample_weights.sum()
            sample_weights = sample_weights / normalization_constant

            # We collect iteration information for later evaluation.
            if self.iterations_to_collect_as_hyperparameters:
                weights = alpha_weights / np.sum(alpha_weights)
                self.collected_weight_vectors_.append(weights.copy())

            loss = zero_one_loss.score(y_neg, np.sign(np.sum(
                np.multiply(classification_matrix,
                            alpha_weights / np.sum(alpha_weights)), axis=1)))
            self.losses.append(loss)

            if self.callback_function is not None:
                self.callback_function(t, alpha_weights, normalization_constant,
                                       self.estimators_generator, self.weights_)

        self.weights_ = alpha_weights / np.sum(alpha_weights)
        self.losses = np.array(self.losses)
        self.learner_info_ = {
            'n_nonzero_weights': np.sum(self.weights_ > 1e-12)}

        return self

    def predict(self, X):
        """Predict inputs using the fit classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to classify.

        Returns
        -------
        predictions : ndarray of shape (n_samples, )
            The estimated labels.

        """
        check_is_fitted(self, 'weights_')
        classification_matrix = self._binary_classification_matrix(X)

        if self.iterations_to_collect_as_hyperparameters:
            self.test_preds = []
            for weight_vector in self.collected_weight_vectors_:
                preds = np.sum(np.multiply(classification_matrix,
                                           weight_vector), axis=1)
                self.test_preds.append(change_label_to_zero(np.sign(preds)))
            self.test_preds = np.array(self.test_preds)
        margins = np.squeeze(
            np.asarray(np.dot(classification_matrix, self.weights_)))
        return change_label_to_zero(
            np.array([int(x) for x in np.sign(margins)]))


class AdaboostGraalpy(AdaBoostGP, BaseMonoviewClassifier):
    """AdaboostGraalpy

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    n_iterations : in number of iterations (default : 200)

    n_stumps : int (default 1)

    kwargs :  others arguments


    Attributes
    ----------
    param_names :

    distribs :

    weird_strings :

    n_stumps :

    nbCores :

    """
    def __init__(self, random_state=None, n_iterations=200, n_stumps=1,
                 **kwargs):

        super(AdaboostGraalpy, self).__init__(
            n_iterations=n_iterations,
            n_stumps=n_stumps
        )
        self.param_names = ["n_iterations", "n_stumps", "random_state"]
        self.distribs = [CustomRandint(low=1, high=500), [n_stumps],
                         [random_state]]
        self.classed_params = []
        self.weird_strings = {}
        self.n_stumps = n_stumps
        if "nbCores" not in kwargs:
            self.nbCores = 1
        else:
            self.nbCores = kwargs["nbCores"]

    # def canProbas(self):
    #     """
    #     Used to know if the classifier can return label probabilities
    #
    #     Returns
    #     -------
    #     True in any case
    #     """
    #     return True

    def getInterpret(self, directory, y_test):
        """

        Parameters
        ----------
        directory :

        y_test :

        Returns
        -------
        retur string of interpret
        """
        np.savetxt(directory + "train_metrics.csv", self.losses, delimiter=',')
        np.savetxt(directory + "y_test_step.csv", self.test_preds,
                   delimiter=',')
        step_metrics = []
        for step_index in range(self.test_preds.shape[0] - 1):
            step_metrics.append(zero_one_loss.score(y_test,
                                                    self.test_preds[step_index,
                                                    :]))
        step_metrics = np.array(step_metrics)
        np.savetxt(directory + "step_test_metrics.csv", step_metrics,
                   delimiter=',')
        return ""


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"n_iterations": args.AdG_n_iter,
#                   "n_stumps": args.AdG_stumps, }
#     return kwargsDict


def paramsToSet(nIter, random_state):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_iterations": random_state.randint(1, 500), })
    return paramsSet

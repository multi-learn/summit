from ..monoview.monoview_utils import CustomUniform, BaseMonoviewClassifier

#### Algorithm code ####

# -*- coding:utf-8 -*-
""" MinCq learning algorithm

Related papers:
[1] From PAC-Bayes Bounds to Quadratic Programs for Majority Votes (Laviolette et al., 2011)
[2] Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm (Germain et al., 2014)

http://graal.ift.ulaval.ca/majorityvote/
"""
__author__ = 'Jean-Francis Roy'
import time
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, \
    polynomial_kernel
# from qp import QP
from ..monoview.additions.BoostUtils import ConvexProgram as QP


classifier_class_name = "MinCQ"

# from majority_vote import MajorityVote
# from voter import StumpsVotersGenerator, KernelVotersGenerator

class MinCqLearner(BaseEstimator, ClassifierMixin):
    """
    MinCq algorithm learner. See [1, 2]

    Parameters
    ----------
    mu : float
        The fixed value of the first moment of the margin.

    voters_type : string, optional (default='kernel')
        Specifies the type of voters.
        It must be one of 'kernel', 'stumps' or 'manual'. If 'manual' is specified, the voters have to be manually set
        using the "voters" parameter of the fit function.

    n_stumps_per_attribute : int, optional (default=10)
        Specifies the amount of decision stumps per attribute.
        It is only significant with 'stumps' voters_type.

    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf'.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default=0.0)
        Kernel coefficient for 'rbf' and 'poly'.
        If gamma is 0.0 then 1/n_features will be used instead.
    """

    def __init__(self, mu, voters_type, n_stumps_per_attribute=10, kernel='rbf',
                 degree=3, gamma=0.0, self_complemented=True):
        assert 0 < mu <= 1, "MinCqLearner: mu parameter must be in (0, 1]"
        self.mu = mu
        self.voters_type = voters_type
        self.n_stumps_per_attribute = n_stumps_per_attribute
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.log = False
        self.self_complemented = self_complemented

        self.majority_vote = None
        self.qp = None

    def fit(self, X, y, voters=None):
        """ Learn a majority vote weights using MinCq.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Training data

        y_reworked : ndarray, shape=(n_samples,), optional
            Training labels

        voters : shape=(n_voters,), optional
            A priori generated voters
        """
        # Preparation of the majority vote, using a voter generator that depends on class attributes
        if (np.unique(y) != [-1, 1]).any():
            y_reworked = np.copy(y)
            y_reworked[np.where(y_reworked == 0)] = -1
        else:
            y_reworked = y

        assert self.voters_type in ['stumps', 'kernel',
                                    'manual'], "MinCqLearner: voters_type must be 'stumps', 'kernel' or 'manual'"

        if self.voters_type == 'manual':
            if voters is None:
                logging.error(
                    "Manually set voters is True, but no voters have been set.")
                return self

        else:
            voters_generator = None

            if self.voters_type == 'stumps':
                assert self.n_stumps_per_attribute >= 1, 'MinCqLearner: n_stumps_per_attribute must be positive'
                voters_generator = StumpsVotersGenerator(
                    self.n_stumps_per_attribute)

            elif self.voters_type == 'kernel':
                assert self.kernel in ['linear', 'poly',
                                       'rbf'], "MinCqLearner: kernel must be 'linear', 'poly' or 'rbf'"

                gamma = self.gamma
                if gamma == 0.0:
                    gamma = 1.0 / np.shape(X)[1]

                if self.kernel == 'linear':
                    voters_generator = KernelVotersGenerator(linear_kernel)
                elif self.kernel == 'poly':
                    voters_generator = KernelVotersGenerator(polynomial_kernel,
                                                             degree=self.degree,
                                                             gamma=gamma)
                elif self.kernel == 'rbf':
                    voters_generator = KernelVotersGenerator(rbf_kernel,
                                                             gamma=gamma)

            voters = voters_generator.generate(X, y_reworked,
                                               self_complemented=self.self_complemented)

        if self.log:
            logging.info("MinCq training started...")
            logging.info("Training dataset shape: {}".format(str(np.shape(X))))
            logging.info("Number of voters: {}".format(len(voters)))
        self.majority_vote = MajorityVote(voters)
        n_base_voters = len(self.majority_vote.weights)

        # Preparation and resolution of the quadratic program

        if self.log:
            logging.info("Preparing QP...")
        self._prepare_qp(X, y_reworked)
        beg = time.time()
        try:
            if self.log:
                logging.info("Solving QP...")
            solver_weights = self.qp.solve()

            # Conversion of the weights of the n first voters to weights on the implicit 2n voters.
            # See Section 7.1 of [2] for an explanation.
            self.majority_vote.weights = np.array(
                [2 * q - 1.0 / n_base_voters for q in solver_weights])
            if self.log:
                logging.info(
                    "First moment of the margin on the training set: {:.4f}".format(
                        np.mean(y_reworked * self.majority_vote.margin(X))))

        except Exception as e:
            logging.error(
                "{}: Error while solving the quadratic program: {}.".format(
                    str(self), str(e)))
            self.majority_vote = None
        self.cbound_train = self.majority_vote.cbound_value(X, y_reworked)
        end=time.time()
        self.train_time=end-beg
        return self

    def predict(self, X, save_data=True):
        """ Using previously learned majority vote weights, predict the labels of new data points.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Samples to predict

        Returns
        -------
        predictions : ndarray, shape=(n_samples,)
            The predicted labels
        """
        if self.log:
            logging.info("Predicting...")
        if self.majority_vote is None:
            logging.error(
                "{}: Error while predicting: MinCq has not been fit or fitting has failed. Will output invalid labels".format(
                    str(self)))
            return np.zeros((len(X),))
        if save_data:
            self.x_test = X

        vote = self.majority_vote.vote(X)
        vote[np.where(vote == -1)] = 0
        return vote

    def predict_proba(self, X):
        """ Using previously learned majority vote weights, predict the labels of new data points with a confidence
        level. The confidence level is the margin of the majority vote.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Samples to predict

        Returns
        -------
        predictions : ndarray, shape=(n_samples,)
            The predicted labels
        """
        probabilities = np.zeros((np.shape(X)[0], 2))

        # The margin is between -1 and 1, we rescale it to be between 0 and 1.
        margins = self.majority_vote.margin(X)
        margins += 1
        margins /= 2

        # Then, the conficence for class +1 is set to the margin, and confidence for class -1 is set to 1 - margin.
        probabilities[:, 1] = margins
        probabilities[:, 0] = 1 - margins
        return probabilities

    def _prepare_qp(self, X, y):
        """ Prepare MinCq's quadratic program. See Program 1 of [2] for more details on its content.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Training data

        y : ndarray, shape=(n_samples,)
            Training labels
        """

        self.qp = QP()

        n_features = len(self.majority_vote.voters)
        n_examples = len(X)
        classification_matrix = self.majority_vote.classification_matrix(X)

        # Objective function.
        self.qp.quadratic_func = 2.0 / n_examples * classification_matrix.T.dot(
            classification_matrix)
        self.qp.linear_func = np.matrix(
            np.matrix(-1.0 * np.mean(self.qp.quadratic_func / 2.0, axis=1))).T

        # First moment of the margin fixed to mu.
        a_matrix = 2.0 / n_examples * y.T.dot(classification_matrix)
        self.qp.add_equality_constraints(a_matrix,
                                         self.mu + 1.0 / 2 * np.mean(a_matrix))

        # Lower and upper bounds on the variables
        self.qp.add_lower_bound(0.0)
        self.qp.add_upper_bound(1.0 / n_features)


class MajorityVote(object):
    """ A Majority Vote of real-valued functions.

    Parameters
    ----------
    voters : ndarray of Voter instances
        The voters of the majority vote. Each voter must take an example as an input, and output a real value in [-1,1].

    weights : ndarray, optional (default: uniform distribution)
        The weights associated to each voter.
    """

    def __init__(self, voters, weights=None):
        self._voters = np.array(voters)

        if weights is not None:
            assert (len(voters) == len(weights))
            self._weights = np.array(weights)
        else:
            self._weights = np.array([1.0 / len(voters)] * len(voters))

    def vote(self, X):
        """ Returns the vote of the Majority Vote on a list of samples.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Input data to classify.

        Returns
        -------
        votes : ndarray, shape=(n_samples,), where each value is either -1 or 1
            The vote of the majority vote for each sample.
        """
        margins = self.margin(X)
        return np.array([int(x) for x in np.sign(margins)])

    def margin(self, X):
        """ Returns the margin of the Majority Vote on a list of samples.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Input data on which to calculate the margin.

        Returns
        -------
        margins : ndarray, shape=(n_samples,), where each value is either -1 or 1
            The margin of the majority vote for each sample.
        """
        classification_matrix = self.classification_matrix(X)
        return np.squeeze(
            np.asarray(np.dot(classification_matrix, self.weights)))

    def classification_matrix(self, X):
        """ Returns the classification matrix of the majority vote.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Input data to classify

        Returns
        -------
        classification_matrix : ndrray, shape=(n_samples, n_voters)
            A matrix that contains the value output by each voter, for each sample.

        """
        return np.matrix([v.vote(X) for v in self._voters]).T

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = np.array(weights)

    @property
    def voters(self):
        return self._voters

    @voters.setter
    def voters(self, voters):
        self._voters = np.array(voters)

    def cbound_value(self, X, y):
        """ Returns the value of the C-bound, evaluated on given examples.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_feature)
            Input data
        y : ndarray, shape=(n_samples, )
            Input labels, where each label is either -1 or 1.
        """
        assert np.all(np.in1d(y, [-1,
                                  1])), 'cbound_value: labels should be either -1 or 1'

        classification_matrix = self.classification_matrix(X)
        first_moment = float(
            1.0 / len(y) * classification_matrix.dot(self.weights).dot(y))
        second_moment = float(1.0 / len(y) * self.weights.T.dot(
            classification_matrix.T.dot(classification_matrix)).dot(
            self.weights))

        return 1 - (first_moment ** 2 / second_moment)


# -*- coding:utf-8 -*-
__author__ = "Jean-Francis Roy"

import numpy as np


class Voter(object):
    """ Base class for a voter (function X -> [-1, 1]), where X is an array of samples
    """

    def __init__(self):
        pass

    def vote(self, X):
        """ Returns the output of the voter, on a sample list X

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Input data to classify

        Returns
        -------
        votes : ndarray, shape=(n_samples,)
            The result the the voter function, for each sample
        """
        raise NotImplementedError("Voter.vote: Not implemented.")


class BinaryKernelVoter(Voter):
    """ A Binary Kernel Voter, which outputs the value of a kernel function whose first example is fixed a priori.
    The sign of the output depends on the label (-1 or 1) of the sample on which the kernel voter is based

    Parameters
    ----------
    x : ndarray, shape=(n_features,)
        The base sample's description vector

    y : int, -1 or 1
        The label of the base sample. Determines if the voter thinks "negative" or "positive"

    kernel_function : function
        The kernel function takes two samples and returns a similarity value. If the kernel has parameters, they should
        be set using kwargs parameter

    kwargs : keyword arguments (optional)
        Additional parameters for the kernel function
    """

    def __init__(self, x, y, kernel_function, **kwargs):
        assert (y in {-1, 1})
        super(BinaryKernelVoter, self).__init__()
        self._x = x
        self._y = y
        self._kernel_function = kernel_function
        self._kernel_kwargs = kwargs

    def vote(self, X):
        base_point_array = np.array([self._x])
        votes = self._y * self._kernel_function(base_point_array, X,
                                                **self._kernel_kwargs)
        votes = np.squeeze(np.asarray(votes))

        return votes


class DecisionStumpVoter(Voter):
    """
    Generic Attribute Threshold Binary Classifier

    Parameters
    ----------
    attribute_index : int
        The attribute to consider for the classification

    threshold : float
        The threshold value for classification rule

    direction : int (-1 or 1)
        Used to reverse classification decision

    Attributes
    ----------

    attribute_index :
    threshold :
    direction :
    """

    def __init__(self, attribute_index, threshold, direction=1):
        super(DecisionStumpVoter, self).__init__()
        self.attribute_index = attribute_index
        self.threshold = threshold
        self.direction = direction

    def vote(self, points):
        return [((point[
                      self.attribute_index] > self.threshold) * 2 - 1) * self.direction
                for point in points]


class VotersGenerator(object):
    """ Base class to create a set of voters using training samples
    """

    def generate(self, X, y=None, self_complemented=False):
        """ Generates the voters using samples.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Input data on which to base the voters

        y : ndarray, shape=(n_samples,), optional
            Input labels, usually determines the decision polarity of each voter

        self_complemented : bool
            Determines if complement voters should be generated or not

        Returns
        -------
        voters : ndarray
            An array of voters
        """
        raise NotImplementedError("VotersGenerator.generate: not implemented")


class StumpsVotersGenerator(VotersGenerator):
    """ Decision Stumps Voters generator.

    Parameters
    ----------
    n_stumps_per_attribute : int, (default=10)
        Determines how many decision stumps will be created for each attribute.
    """

    def __init__(self, n_stumps_per_attribute=10):
        self._n_stumps_per_attribute = n_stumps_per_attribute

    def _find_extremums(self, X, i):
        mini = np.Infinity
        maxi = -np.Infinity
        for x in X:
            if x[i] < mini:
                mini = x[i]
            if x[i] > maxi:
                maxi = x[i]
        return mini, maxi

    def generate(self, X, y=None, self_complemented=False,
                 only_complements=False):
        """

        Parameters
        ----------
        X
        y
        self_complemented
        only_complements

        Returns
        -------

        """
        voters = []
        if len(X) != 0:
            for i in range(len(X[0])):
                t = self._find_extremums(X, i)
                inter = (t[1] - t[0]) / (self._n_stumps_per_attribute + 1)

                if inter != 0:
                    # If inter is zero, the attribute is useless as it has a constant value. We do not add stumps for
                    # this attribute.
                    for x in range(self._n_stumps_per_attribute):

                        if not only_complements:
                            voters.append(
                                DecisionStumpVoter(i, t[0] + inter * (x + 1),
                                                   1))

                        if self_complemented or only_complements:
                            voters.append(
                                DecisionStumpVoter(i, t[0] + inter * (x + 1),
                                                   -1))

        return np.array(voters)


class KernelVotersGenerator(VotersGenerator):
    """ Utility function to create binary kernel voters for each (x, y) sample.

    Parameters
    ----------
    kernel_function : function
        The kernel function takes two samples and returns a similarity value. If the kernel has parameters, they should
        be set using kwargs parameter

    kwargs : keyword arguments (optional)
        Additional parameters for the kernel function
    """

    def __init__(self, kernel_function, **kwargs):
        self._kernel_function = kernel_function
        self._kernel_kwargs = kwargs

    def generate(self, X, y=None, self_complemented=False,
                 only_complements=False):
        if y is None:
            y = np.array([1] * len(X))

        voters = []

        for point, label in zip(X, y):
            if not only_complements:
                voters.append(
                    BinaryKernelVoter(point, label, self._kernel_function,
                                      **self._kernel_kwargs))

            if self_complemented or only_complements:
                voters.append(
                    BinaryKernelVoter(point, -1 * label, self._kernel_function,
                                      **self._kernel_kwargs))

        return np.array(voters)


class MinCQ(MinCqLearner, BaseMonoviewClassifier):

    def __init__(self, random_state=None, mu=0.01, self_complemented=True,
                 n_stumps_per_attribute=10, **kwargs):
        super(MinCQ, self).__init__(mu=mu,
                                    voters_type='stumps',
                                    n_stumps_per_attribute=n_stumps_per_attribute,
                                    self_complemented=self_complemented
                                    )
        self.param_names = ["mu", "n_stumps_per_attribute", "random_state"]
        self.distribs = [CustomUniform(loc=0.5, state=2.0, multiplier="e-"),
                         [n_stumps_per_attribute], [random_state]]
        self.random_state = random_state
        self.classed_params = []
        self.weird_strings = {}
        if "nbCores" not in kwargs:
            self.nbCores = 1
        else:
            self.nbCores = kwargs["nbCores"]

    # def canProbas(self):
    #     """Used to know if the classifier can return label probabilities"""
    #     return True

    def set_params(self, **params):
        self.mu = params["mu"]
        self.random_state = params["random_state"]
        self.n_stumps_per_attribute = params["n_stumps_per_attribute"]
        return self

    def get_params(self, deep=True):
        return {"random_state": self.random_state, "mu": self.mu,
                "n_stumps_per_attribute": self.n_stumps_per_attribute}

    def getInterpret(self, directory, y_test):
        interpret_string = "Train C_bound value : " + str(self.cbound_train)
        y_rework = np.copy(y_test)
        y_rework[np.where(y_rework == 0)] = -1
        interpret_string += "\n Test c_bound value : " + str(
            self.majority_vote.cbound_value(self.x_test, y_rework))
        np.savetxt(directory+"times.csv", np.array([self.train_time, 0]))
        return interpret_string

    def get_name_for_fusion(self):
        return "MCQ"

#
# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"mu": args.MCQ_mu,
#                   "n_stumps_per_attribute": args.MCQ_stumps}
#     return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet

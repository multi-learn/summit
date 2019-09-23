import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


class DecisionStumpClassifier(BaseEstimator, ClassifierMixin):
    """Generic Attribute Threshold Binary Classifier

    Attributes
    ----------
    attribute_index : int
        The attribute to consider for the classification.
    threshold : float
        The threshold value for classification rule.
    direction : int, optional
        A multiplicative constant (1 or -1) to choose the "direction" of the stump. Defaults to 1. If -1, the stump
        will predict the "negative" class (generally -1 or 0), and if 1, the stump will predict the second class (generally 1).

    """

    def __init__(self, attribute_index, threshold, direction=1):
        super(DecisionStumpClassifier, self).__init__()
        self.attribute_index = attribute_index
        self.threshold = threshold
        self.direction = direction

    def fit(self, X, y):
        # Only verify that we are in the binary classification setting, with support for transductive learning.
        if isinstance(y, np.ma.MaskedArray):
            self.classes_ = np.unique(y[np.logical_not(y.mask)])
        else:
            self.classes_ = np.unique(y)

        # This label encoder is there for the predict function to be able to return any two classes that were used
        # when fitting, for example {-1, 1} or {0, 1}.
        self.le_ = LabelEncoder()
        self.le_.fit(self.classes_)
        self.classes_ = self.le_.classes_

        if not len(self.classes_) == 2:
            raise ValueError(
                'DecisionStumpsVoter only supports binary classification')
        # assert len(self.classes_) == 2, "DecisionStumpsVoter only supports binary classification"
        return self

    def predict(self, X):
        """Returns the output of the classifier, on a sample X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        predictions : array-like, shape = [n_samples]
            Predicted class labels.

        """
        check_is_fitted(self, 'classes_')
        return self.le_.inverse_transform(
            np.argmax(self.predict_proba(X), axis=1))

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """
        check_is_fitted(self, 'classes_')
        X = np.asarray(X)
        probas = np.zeros((X.shape[0], 2))
        positive_class = np.argwhere(
            X[:, self.attribute_index] > self.threshold)
        negative_class = np.setdiff1d(range(X.shape[0]), positive_class)
        probas[positive_class, 1] = 1.0
        probas[negative_class, 0] = 1.0

        if self.direction == -1:
            probas = 1 - probas

        return probas

    def predict_proba_t(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """

        X = np.ones(X.shape)
        check_is_fitted(self, 'classes_')
        X = np.asarray(X)
        probas = np.zeros((X.shape[0], 2))
        positive_class = np.argwhere(
            X[:, self.attribute_index] > self.threshold)
        negative_class = np.setdiff1d(range(X.shape[0]), positive_class)
        probas[positive_class, 1] = 1.0
        probas[negative_class, 0] = 1.0

        if self.direction == -1:
            probas = 1 - probas

        return probas

    def reverse_decision(self):
        self.direction *= -1


class ClassifiersGenerator(BaseEstimator, TransformerMixin):
    """Base class to create a set of voters using training samples, and then transform a set of examples in
    the voters' output space.

    Attributes
    ----------
    self_complemented : bool, optional
        Whether or not a binary complement voter must be generated for each voter. Defaults to False.
    voters : ndarray of voter functions
        Once fit, contains the voter functions.

    """

    def __init__(self, self_complemented=False):
        super(ClassifiersGenerator, self).__init__()
        self.self_complemented = self_complemented

    def fit(self, X, y=None):
        """Generates the voters using training samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data on which to base the voters.
        y : ndarray of shape (n_labeled_samples,), optional
            Input labels, usually determines the decision polarity of each voter.

        Returns
        -------
        self

        """
        raise NotImplementedError

    def transform(self, X):
        """Transforms the input points in a matrix of classification, using previously learned voters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to classify.

        Returns
        -------
        ndarray of shape (n_samples, n_voters)
            The voters' decision on each example.

        """
        check_is_fitted(self, 'estimators_')
        return np.array([voter.predict(X) for voter in self.estimators_]).T


# class TreesClassifiersGenerator(ClassifiersGenerator):
#     """A generator to widen the voter's pool of our boosting algorithms.
#     """
#
#     def __init__(self, n_stumps_per_attribute=10, self_complemented=False, check_diff=True, max_depth=3):
#         super(TreesClassifiersGenerator, self).__init__(self_complemented)
#         self.n_stumps_per_attribute = n_stumps_per_attribute
#         self.check_diff = check_diff
#         self.max_depth = max_depth
#
#     def fit(self, X, y=None):

class TreeClassifiersGenerator(ClassifiersGenerator):

    def __init__(self, random_state=42, max_depth=2, self_complemented=True,
                 criterion="gini", splitter="best", n_trees=100,
                 distribution_type="uniform", low=0, high=10,
                 attributes_ratio=0.6, examples_ratio=0.95):
        super(TreeClassifiersGenerator, self).__init__(self_complemented)
        self.max_depth = max_depth
        self.criterion = criterion
        self.splitter = splitter
        self.n_trees = n_trees
        if type(random_state) is int:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
        self.distribution_type = distribution_type
        self.low = low
        self.high = high
        self.attributes_ratio = attributes_ratio
        self.examples_ratio = examples_ratio

    def fit(self, X, y=None):
        estimators_ = []
        self.attribute_indices = np.array(
            [self.sub_sample_attributes(X) for _ in range(self.n_trees)])
        self.example_indices = np.array(
            [self.sub_sample_examples(X) for _ in range(self.n_trees)])
        for i in range(self.n_trees):
            estimators_.append(DecisionTreeClassifier(criterion=self.criterion,
                                                      splitter=self.splitter,
                                                      max_depth=self.max_depth).fit(
                X[:, self.attribute_indices[i, :]][self.example_indices[i], :],
                y[self.example_indices[i, :]]))
        self.estimators_ = np.asarray(estimators_)
        return self

    def sub_sample_attributes(self, X):
        n_attributes = X.shape[1]
        attributes_indices = np.arange(n_attributes)
        kept_indices = self.random_state.choice(attributes_indices, size=int(
            self.attributes_ratio * n_attributes), replace=True)
        return kept_indices

    def sub_sample_examples(self, X):
        n_examples = X.shape[0]
        examples_indices = np.arange(n_examples)
        kept_indices = self.random_state.choice(examples_indices, size=int(
            self.examples_ratio * n_examples), replace=True)
        return kept_indices

    def choose(self, chosen_columns):
        self.estimators_ = self.estimators_[chosen_columns]
        self.attribute_indices = self.attribute_indices[chosen_columns, :]
        self.example_indices = self.example_indices[chosen_columns, :]


class StumpsClassifiersGenerator(ClassifiersGenerator):
    """Decision Stump Voters transformer.

    Parameters
    ----------
    n_stumps_per_attribute : int, optional
        Determines how many decision stumps will be created for each attribute. Defaults to 10.
        No stumps will be created for attributes with only one possible value.
    self_complemented : bool, optional
        Whether or not a binary complement voter must be generated for each voter. Defaults to False.

    """

    def __init__(self, n_stumps_per_attribute=10, self_complemented=False,
                 check_diff=False):
        super(StumpsClassifiersGenerator, self).__init__(self_complemented)
        self.n_stumps_per_attribute = n_stumps_per_attribute
        self.check_diff = check_diff

    def fit(self, X, y=None):
        """Fits Decision Stump voters on a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data on which to base the voters.
        y : ndarray of shape (n_labeled_samples,), optional
            Only used to ensure that we are in the binary classification setting.

        Returns
        -------
        self

        """
        minimums = np.min(X, axis=0)
        maximums = np.max(X, axis=0)
        if y.ndim > 1:
            y = np.reshape(y, (y.shape[0],))
        ranges = (maximums - minimums) / (self.n_stumps_per_attribute + 1)
        if self.check_diff:
            nb_differents = [np.unique(col) for col in np.transpose(X)]
            self.estimators_ = []
            for i in range(X.shape[1]):
                nb_different = nb_differents[i].shape[0]
                different = nb_differents[i]
                if nb_different - 1 < self.n_stumps_per_attribute:
                    self.estimators_ += [DecisionStumpClassifier(i,
                                                                 (different[
                                                                      stump_number] +
                                                                  different[
                                                                      stump_number + 1]) / 2,
                                                                 1).fit(X, y)
                                         for stump_number in
                                         range(int(nb_different) - 1)]
                    if self.self_complemented:
                        self.estimators_ += [DecisionStumpClassifier(i,
                                                                     (different[
                                                                          stump_number] +
                                                                      different[
                                                                          stump_number + 1]) / 2,
                                                                     -1).fit(X,
                                                                             y)
                                             for stump_number in
                                             range(int(nb_different) - 1)]
                else:
                    self.estimators_ += [DecisionStumpClassifier(i,
                                                                 minimums[i] +
                                                                 ranges[
                                                                     i] * stump_number,
                                                                 1).fit(X, y)
                                         for stump_number in range(1,
                                                                   self.n_stumps_per_attribute + 1)
                                         if ranges[i] != 0]

                    if self.self_complemented:
                        self.estimators_ += [DecisionStumpClassifier(i,
                                                                     minimums[
                                                                         i] +
                                                                     ranges[
                                                                         i] * stump_number,
                                                                     -1).fit(X,
                                                                             y)
                                             for stump_number in range(1,
                                                                       self.n_stumps_per_attribute + 1)
                                             if ranges[i] != 0]
        else:
            self.estimators_ = [DecisionStumpClassifier(i, minimums[i] + ranges[
                i] * stump_number, 1).fit(X, y)
                                for i in range(X.shape[1]) for stump_number in
                                range(1, self.n_stumps_per_attribute + 1)
                                if ranges[i] != 0]

            if self.self_complemented:
                self.estimators_ += [DecisionStumpClassifier(i, minimums[i] +
                                                             ranges[
                                                                 i] * stump_number,
                                                             -1).fit(X, y)
                                     for i in range(X.shape[1]) for stump_number
                                     in
                                     range(1, self.n_stumps_per_attribute + 1)
                                     if ranges[i] != 0]
        self.estimators_ = np.asarray(self.estimators_)
        return self

    def choose(self, chosen_columns):
        self.estimators_ = self.estimators_[chosen_columns]


def _as_matrix(element):
    """ Utility function to convert "anything" to a Numpy matrix.
    """
    # If a scalar, return a 1x1 matrix.
    if len(np.shape(element)) == 0:
        return np.matrix([[element]], dtype=float)

    # If a nd-array vector, return a column matrix.
    elif len(np.shape(element)) == 1:
        matrix = np.matrix(element, dtype=float)
        if np.shape(matrix)[1] != 1:
            matrix = matrix.T
        return matrix

    return np.matrix(element, dtype=float)


def _as_column_matrix(array_like):
    """ Utility function to convert any array to a column Numpy matrix.
    """
    matrix = _as_matrix(array_like)
    if 1 not in np.shape(matrix):
        raise ValueError("_as_column_vector: input must be a vector")

    if np.shape(matrix)[0] == 1:
        matrix = matrix.T

    return matrix


def _as_line_matrix(array_like):
    """ Utility function to convert any array to a line Numpy matrix.
    """
    matrix = _as_matrix(array_like)
    if 1 not in np.shape(matrix):
        raise ValueError("_as_column_vector: input must be a vector")

    if np.shape(matrix)[1] == 1:
        matrix = matrix.T

    return matrix


def sign(array):
    """Computes the elementwise sign of all elements of an array. The sign function returns -1 if x <=0 and 1 if x > 0.
    Note that numpy's sign function can return 0, which is not desirable in most cases in Machine Learning algorithms.

    Parameters
    ----------
    array : array-like
        Input values.

    Returns
    -------
    ndarray
        An array with the signs of input elements.

    """
    signs = np.sign(array)

    signs[array == 0] = -1
    return signs


class ConvexProgram(object):
    """
    Encapsulates a quadratic program of the following form:

    minimize    (1/2)*x'*P*x + q'*x
    subject to  G*x <= h
                A*x = b.


    or a linear program of the following form:

    minimize    c'*x
    subject to  G*x <= h
                A*x = b
    """

    def __init__(self):
        self._quadratic_func = None
        self._linear_func = None
        self._inequality_constraints_matrix = None
        self._inequality_constraints_values = None
        self._equality_constraints_matrix = None
        self._equality_constraints_values = None
        self._lower_bound_values = None
        self._upper_bound_values = None
        self._n_variables = None

    @property
    def n_variables(self):
        return self._n_variables

    @property
    def quadratic_func(self):
        return self._quadratic_func

    @quadratic_func.setter
    def quadratic_func(self, quad_matrix):
        quad_matrix = _as_matrix(quad_matrix)
        n_lines, n_columns = np.shape(quad_matrix)
        assert (n_lines == n_columns)

        if self._linear_func is not None:
            assert (np.shape(quad_matrix)[0] == self._n_variables)
        else:
            self._n_variables = n_lines

        self._quadratic_func = quad_matrix

    @property
    def linear_func(self):
        return self._linear_func

    @linear_func.setter
    def linear_func(self, lin_vector):
        if lin_vector is not None:
            lin_vector = _as_column_matrix(lin_vector)

            if self._quadratic_func is not None:
                assert (np.shape(lin_vector)[0] == self._n_variables)

            else:
                self._n_variables = np.shape(lin_vector)[0]

            self._linear_func = lin_vector

    def add_inequality_constraints(self, inequality_matrix, inequality_values):
        if inequality_matrix is None:
            return

        self._assert_objective_function_is_set()

        if 1 in np.shape(inequality_matrix) or len(
                np.shape(inequality_matrix)) == 1:
            inequality_matrix = _as_line_matrix(inequality_matrix)
        else:
            inequality_matrix = _as_matrix(inequality_matrix)

        inequality_values = _as_column_matrix(inequality_values)
        assert np.shape(inequality_matrix)[1] == self._n_variables
        assert np.shape(inequality_values)[1] == 1

        if self._inequality_constraints_matrix is None:
            self._inequality_constraints_matrix = inequality_matrix
        else:
            self._inequality_constraints_matrix = np.append(
                self._inequality_constraints_matrix,
                inequality_matrix, axis=0)

        if self._inequality_constraints_values is None:
            self._inequality_constraints_values = inequality_values
        else:
            self._inequality_constraints_values = np.append(
                self._inequality_constraints_values,
                inequality_values, axis=0)

    def add_equality_constraints(self, equality_matrix, equality_values):
        if equality_matrix is None:
            return

        self._assert_objective_function_is_set()

        if 1 in np.shape(equality_matrix) or len(
                np.shape(equality_matrix)) == 1:
            equality_matrix = _as_line_matrix(equality_matrix)
        else:
            equality_matrix = _as_matrix(equality_matrix)

        equality_values = _as_matrix(equality_values)
        assert np.shape(equality_matrix)[1] == self._n_variables
        assert np.shape(equality_values)[1] == 1

        if self._equality_constraints_matrix is None:
            self._equality_constraints_matrix = equality_matrix
        else:
            self._equality_constraints_matrix = np.append(
                self._equality_constraints_matrix,
                equality_matrix, axis=0)

        if self._equality_constraints_values is None:
            self._equality_constraints_values = equality_values
        else:
            self._equality_constraints_values = np.append(
                self._equality_constraints_values,
                equality_values, axis=0)

    def add_lower_bound(self, lower_bound):
        if lower_bound is not None:
            self._assert_objective_function_is_set()
            self._lower_bound_values = np.array(
                [lower_bound] * self._n_variables)

    def add_upper_bound(self, upper_bound):
        if upper_bound is not None:
            self._assert_objective_function_is_set()
            self._upper_bound_values = np.array(
                [upper_bound] * self._n_variables)

    def _convert_bounds_to_inequality_constraints(self):
        self._assert_objective_function_is_set()

        if self._lower_bound_values is not None:
            c_matrix = []
            for i in range(self._n_variables):
                c_line = [0] * self._n_variables
                c_line[i] = -1.0
                c_matrix.append(c_line)

            c_vector = _as_column_matrix(self._lower_bound_values)
            self._lower_bound_values = None
            self.add_inequality_constraints(np.matrix(c_matrix).T, c_vector)

        if self._upper_bound_values is not None:
            c_matrix = []
            for i in range(self._n_variables):
                c_line = [0] * self._n_variables
                c_line[i] = 1.0
                c_matrix.append(c_line)

            c_vector = _as_column_matrix(self._upper_bound_values)
            self._upper_bound_values = None
            self.add_inequality_constraints(np.matrix(c_matrix).T, c_vector)

    def _convert_to_cvxopt_matrices(self):
        from cvxopt import matrix as cvxopt_matrix

        if self._quadratic_func is not None:
            self._quadratic_func = cvxopt_matrix(self._quadratic_func)

        if self._linear_func is not None:
            self._linear_func = cvxopt_matrix(self._linear_func)
        else:
            # CVXOPT needs this vector to be set even if it is not used, so we put zeros in it!
            self._linear_func = cvxopt_matrix(np.zeros((self._n_variables, 1)))

        if self._inequality_constraints_matrix is not None:
            self._inequality_constraints_matrix = cvxopt_matrix(
                self._inequality_constraints_matrix)

        if self._inequality_constraints_values is not None:
            self._inequality_constraints_values = cvxopt_matrix(
                self._inequality_constraints_values)

        if self._equality_constraints_matrix is not None:
            self._equality_constraints_matrix = cvxopt_matrix(
                self._equality_constraints_matrix)

        if self._equality_constraints_values is not None:
            self._equality_constraints_values = cvxopt_matrix(
                self._equality_constraints_values)

    def _assert_objective_function_is_set(self):
        assert self._n_variables is not None

    def solve(self, solver="cvxopt", feastol=1e-7, abstol=1e-7, reltol=1e-6,
              return_all_information=False):

        # Some solvers are very verbose, and we don't want them to pollute STDOUT or STDERR.
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        ret = None

        # TODO: Repair
        # if solver == "cvxopt":
        #     stdout_logger = logging.getLogger('CVXOPT')
        #     sl = StreamToLogger(stdout_logger, logging.DEBUG)
        #     sys.stdout = sl

        #     stderr_logger = logging.getLogger('CVXOPT')
        #     sl = StreamToLogger(stderr_logger, logging.WARNING)
        #     sys.stderr = sl

        try:
            if solver == "cvxopt":
                from cvxopt.solvers import qp, lp, options
                options['feastol'] = feastol
                options['abstol'] = abstol
                options['reltol'] = reltol
                options['show_progress'] = False

                self._convert_bounds_to_inequality_constraints()
                self._convert_to_cvxopt_matrices()

                if self._quadratic_func is not None:
                    ret = qp(self.quadratic_func, self.linear_func,
                             self._inequality_constraints_matrix,
                             self._inequality_constraints_values,
                             self._equality_constraints_matrix,
                             self._equality_constraints_values)

                else:
                    ret = lp(self.linear_func,
                             G=self._inequality_constraints_matrix,
                             h=self._inequality_constraints_values,
                             A=self._equality_constraints_matrix,
                             b=self._equality_constraints_values)

                # logging.info("Primal objective value  = {}".format(ret['primal objective']))
                # logging.info("Dual objective value  = {}".format(ret['dual objective']))

                if not return_all_information:
                    ret = np.asarray(np.array(ret['x']).T[0])

            elif solver == "cplex":
                import cplex
                p = cplex.Cplex()
                p.objective.set_sense(p.objective.sense.minimize)

                # This is ugly. CPLEX wants a list of lists of lists. First dimension represents the lines of the QP
                # matrix. Second dimension contains a pair of two elements: the indices of the variables in play (all of
                # them...), and the values (columns of the QP matrix).
                names = [str(x) for x in range(self._n_variables)]
                p.variables.add(names=names)

                if self.quadratic_func is not None:
                    p_matrix = []
                    for line in self._quadratic_func:
                        p_matrix.append([names, line.tolist()[0]])

                    p.objective.set_quadratic(p_matrix)

                if self.linear_func is not None:
                    p.objective.set_linear(zip(names,
                                               np.asarray(
                                                   self.linear_func.T).reshape(
                                                   self.n_variables, ).tolist()))

                if self._inequality_constraints_matrix is not None:
                    inequality_linear = []
                    for line in self._inequality_constraints_matrix:
                        inequality_linear.append([names, line.tolist()[0]])
                    p.linear_constraints.add(lin_expr=inequality_linear,
                                             rhs=np.asarray(
                                                 self._inequality_constraints_values.T).tolist()[
                                                 0],
                                             senses="L" * len(
                                                 self._inequality_constraints_values))

                if self._equality_constraints_matrix is not None:
                    equality_linear = []
                    for line in self._equality_constraints_matrix:
                        equality_linear.append([names, line.tolist()[0]])
                    p.linear_constraints.add(lin_expr=equality_linear,
                                             rhs=np.asarray(
                                                 self._equality_constraints_values.T).tolist()[
                                                 0],
                                             senses="E" * len(
                                                 self._equality_constraints_values))

                if self._lower_bound_values is not None:
                    p.variables.set_lower_bounds(
                        zip(names, self._lower_bound_values))

                if self._upper_bound_values is not None:
                    p.variables.set_upper_bounds(
                        zip(names, self._upper_bound_values))

                p.solve()

                if not return_all_information:
                    ret = np.array(p.solution.get_values())
                else:
                    ret = {'primal': np.array(p.solution.get_values()),
                           'dual': np.array(p.solution.get_dual_values())}

            elif solver == "pycpx":
                # This shows how easy it is to use pycpx. However, it is much slower (as it is more versatile!).

                import pycpx
                model = pycpx.CPlexModel(verbosity=2)
                q = model.new(self.n_variables)

                if self._inequality_constraints_matrix is not None:
                    model.constrain(
                        self._inequality_constraints_matrix * q <= self._inequality_constraints_values)
                if self._equality_constraints_matrix is not None:
                    model.constrain(
                        self._equality_constraints_matrix * q == self._equality_constraints_values)
                if self._lower_bound_values is not None:
                    model.constrain(q >= self._lower_bound_values)
                if self._upper_bound_values is not None:
                    model.constrain(q <= self._upper_bound_values)

                value = model.minimize(
                    0.5 * q.T * self._quadratic_func * q + self.linear_func.T * q)

                if not return_all_information:
                    ret = np.array(model[q])
                else:
                    ret = model

        except:
            raise

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        return ret

    def _as_matrix(element):
        """ Utility function to convert "anything" to a Numpy matrix.
        """
        # If a scalar, return a 1x1 matrix.
        if len(np.shape(element)) == 0:
            return np.matrix([[element]], dtype=float)

        # If a nd-array vector, return a column matrix.
        elif len(np.shape(element)) == 1:
            matrix = np.matrix(element, dtype=float)
            if np.shape(matrix)[1] != 1:
                matrix = matrix.T
            return matrix

        return np.matrix(element, dtype=float)

    def _as_column_matrix(array_like):
        """ Utility function to convert any array to a column Numpy matrix.
        """
        matrix = _as_matrix(array_like)
        if 1 not in np.shape(matrix):
            raise ValueError("_as_column_vector: input must be a vector")

        if np.shape(matrix)[0] == 1:
            matrix = matrix.T

        return matrix

    def _as_line_matrix(array_like):
        """ Utility function to convert any array to a line Numpy matrix.
        """
        matrix = _as_matrix(array_like)
        if 1 not in np.shape(matrix):
            raise ValueError("_as_column_vector: input must be a vector")

        if np.shape(matrix)[1] == 1:
            matrix = matrix.T

        return matrix

    def sign(array):
        """Computes the elementwise sign of all elements of an array. The sign function returns -1 if x <=0 and 1 if x > 0.
        Note that numpy's sign function can return 0, which is not desirable in most cases in Machine Learning algorithms.

        Parameters
        ----------
        array : array-like
            Input values.

        Returns
        -------
        ndarray
            An array with the signs of input elements.

        """
        signs = np.sign(array)

        signs[array == 0] = -1
        return signs


def get_accuracy_graph(plotted_data, classifier_name, file_name,
                       name="Accuracies", bounds=None, bound_name=None,
                       boosting_bound=None, set="train", zero_to_one=True):
    if type(name) is not str:
        name = " ".join(name.getConfig().strip().split(" ")[:2])
    f, ax = plt.subplots(nrows=1, ncols=1)
    if zero_to_one:
        ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_title(name + " during " + set + " for " + classifier_name)
    x = np.arange(len(plotted_data))
    scat = ax.scatter(x, np.array(plotted_data), marker=".")
    if bounds:
        if boosting_bound:
            scat2 = ax.scatter(x, boosting_bound, marker=".")
            scat3 = ax.scatter(x, np.array(bounds), marker=".", )
            ax.legend((scat, scat2, scat3),
                      (name, "Boosting bound", bound_name))
        else:
            scat2 = ax.scatter(x, np.array(bounds), marker=".", )
            ax.legend((scat, scat2),
                      (name, bound_name))
        # plt.tight_layout()
    else:
        ax.legend((scat,), (name,))
    f.savefig(file_name, transparent=True)
    plt.close()


class BaseBoost(object):

    def _collect_probas(self, X, sub_sampled=False):
        if self.estimators_generator.__class__.__name__ == "TreeClassifiersGenerator":
            return np.asarray([clf.predict_proba(X[:, attribute_indices]) for
                               clf, attribute_indices in
                               zip(self.estimators_generator.estimators_,
                                   self.estimators_generator.attribute_indices)])
        else:
            return np.asarray([clf.predict_proba(X) for clf in
                               self.estimators_generator.estimators_])

    def _binary_classification_matrix(self, X):
        probas = self._collect_probas(X)
        predicted_labels = np.argmax(probas, axis=2)
        predicted_labels[predicted_labels == 0] = -1
        values = np.max(probas, axis=2)
        return (predicted_labels * values).T

    def _initialize_alphas(self, n_examples):
        raise NotImplementedError(
            "Alpha weights initialization function is not implemented.")

    def check_opposed_voters(self, ):
        nb_opposed = 0
        oppposed = []
        for column in self.classification_matrix[:,
                      self.chosen_columns_].transpose():
            for chosen_col in self.chosen_columns_:
                if (-column.reshape((self.n_total_examples,
                                     1)) == self.classification_matrix[:,
                                            chosen_col].reshape(
                        (self.n_total_examples, 1))).all():
                    nb_opposed += 1
                    break
        return int(nb_opposed / 2)


def getInterpretBase(classifier, directory, classifier_name, weights,
                     break_cause=" the dual constrail was not violated"):
    interpretString = "\t " + classifier_name + " permformed classification with weights : \n"
    # weights_sort = np.argsort(-weights)
    weights_sort = np.arange(weights.shape[0])
    interpretString += np.array2string(weights[weights_sort], precision=4,
                                       separator=',', suppress_small=True)
    interpretString += "\n \t It generated {} columns by attributes and used {} iterations to converge, and selected {} couple(s) of opposed voters".format(
        classifier.n_stumps,
        len(weights_sort), classifier.nb_opposed_voters)
    if max(weights) > 0.50:
        interpretString += "\n \t The vote is useless in this context : voter nb {} is a dictator of weight > 0.50".format(
            classifier.chosen_columns_[np.argmax(np.array(weights))])
    if len(weights_sort) == classifier.n_max_iterations or len(
            weights) == classifier.n_total_hypotheses_:
        if len(weights) == classifier.n_max_iterations:
            interpretString += ", and used all available iterations, "
        else:
            interpretString += "."
        if len(weights) == classifier.n_total_hypotheses_:
            interpretString += ", and all the voters have been used."
        else:
            interpretString += "."
    else:
        pass
        # interpretString += ", and the loop was broken because "+break_cause
    interpretString += "\n\t Selected voters : \n"
    interpretString += np.array2string(
        np.array(classifier.chosen_columns_)[weights_sort])
    interpretString += "\n\t Trained in " + str(datetime.timedelta(
        seconds=classifier.train_time)) + " and predicted in " + str(
        datetime.timedelta(seconds=classifier.predict_time)) + "."
    interpretString += "\n\t Selected columns : \n"
    interpretString += np.array2string(
        classifier.classification_matrix[:, classifier.chosen_columns_],
        precision=4,
        separator=',', suppress_small=True)
    np.savetxt(directory + "voters.csv",
               classifier.classification_matrix[:, classifier.chosen_columns_],
               delimiter=',')
    np.savetxt(directory + "weights.csv", classifier.weights_, delimiter=',')
    np.savetxt(directory + "times.csv",
               np.array([classifier.train_time, classifier.predict_time]),
               delimiter=',')
    np.savetxt(directory + "times_iter.csv",
               np.array([classifier.train_time, len(weights_sort)]),
               delimiter=',')
    np.savetxt(directory + "sparsity.csv", np.array([len(weights_sort)]),
               delimiter=',')
    get_accuracy_graph(classifier.train_metrics, classifier_name,
                       directory + 'metrics.png', classifier.plotted_metric,
                       classifier.bounds, "Boosting bound")
    return interpretString

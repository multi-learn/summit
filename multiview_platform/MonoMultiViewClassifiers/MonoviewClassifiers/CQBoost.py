import scipy
import logging
from future.utils import iteritems
from copy import deepcopy
import numpy.ma as ma
from collections import defaultdict, OrderedDict
import pandas as pd
import sys
from functools import partial
import numpy as np
from scipy.spatial import distance
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint, uniform
import numpy as np

from ..Monoview.MonoviewUtils import CustomUniform, CustomRandint



class ColumnGenerationClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epsilon=1e-06, n_max_iterations=None, estimators_generator=None, dual_constraint_rhs=0, save_iteration_as_hyperparameter_each=None):
        self.epsilon = epsilon
        self.n_max_iterations = n_max_iterations
        self.estimators_generator = estimators_generator
        self.dual_constraint_rhs = dual_constraint_rhs
        self.save_iteration_as_hyperparameter_each = save_iteration_as_hyperparameter_each

    def fit(self, X, y):
        if scipy.sparse.issparse(X):
            # logging.info('Converting to dense matrix.')
            X = np.array(X.todense())

        y[y == 0] = -1

        if self.estimators_generator is None:
            self.estimators_generator = StumpsClassifiersGenerator(n_stumps_per_attribute=10, self_complemented=True)

        self.estimators_generator.fit(X, y)
        self.classification_matrix = self._binary_classification_matrix(X)

        self.infos_per_iteration_ = defaultdict(list)

        m, n = self.classification_matrix.shape
        # self.chosen_columns_ = [np.random.choice(np.arange(n)), np.random.choice(np.arange(n))]
        self.chosen_columns_ = []
        self.n_total_hypotheses_ = n

        y_kernel_matrix = np.multiply(y.reshape((len(y), 1)), self.classification_matrix)

        # Initialization
        alpha = self._initialize_alphas(m)
        # w = [0.5,0.5]
        w= None
        self.collected_weight_vectors_ = {}
        self.collected_dual_constraint_violations_ = {}

        for k in range(min(n, self.n_max_iterations if self.n_max_iterations is not None else np.inf)):
            # Find worst weak hypothesis given alpha.
            h_values = ma.array(np.squeeze(np.array((alpha).T.dot(y_kernel_matrix).T)), fill_value=-np.inf)
            h_values[self.chosen_columns_] = ma.masked
            worst_h_index = ma.argmax(h_values)
            # logging.info("Adding voter {} to the columns, value = {}".format(worst_h_index, h_values[worst_h_index]))

            # Check for optimal solution. We ensure at least one complete iteration is done as the initialization
            # values might provide a degenerate initial solution.
            if h_values[worst_h_index] <= self.dual_constraint_rhs + self.epsilon and len(self.chosen_columns_) > 0:
                break

            # Append the weak hypothesis.
            self.chosen_columns_.append(worst_h_index)

            # Solve restricted master for new costs.
            w, alpha = self._restricted_master_problem(y_kernel_matrix[:, self.chosen_columns_], previous_w=w, previous_alpha=alpha)


            # We collect iteration information for later evaluation.
            if self.save_iteration_as_hyperparameter_each is not None:
                if (k + 1) % self.save_iteration_as_hyperparameter_each == 0:
                    self.collected_weight_vectors_[k] = deepcopy(w)
                    self.collected_dual_constraint_violations_[k] = h_values[worst_h_index] - self.dual_constraint_rhs

        self.weights_ = w
        self.estimators_generator.estimators_ = self.estimators_generator.estimators_[self.chosen_columns_]

        self.learner_info_ = {}
        self.learner_info_.update(n_nonzero_weights=np.sum(np.asarray(self.weights_) > 1e-12))
        self.learner_info_.update(n_generated_columns=len(self.chosen_columns_))
        y[y == -1] = 0
        return self

    def predict(self, X):
        check_is_fitted(self, 'weights_')

        if scipy.sparse.issparse(X):
            logging.warning('Converting sparse matrix to dense matrix.')
            X = np.array(X.todense())

        classification_matrix = self._binary_classification_matrix(X)

        margins = np.squeeze(np.asarray(np.dot(classification_matrix, self.weights_)))
        signs_array = np.array([int(x) for x in sign(margins)])
        signs_array[signs_array == -1] = 0
        return signs_array

    def _binary_classification_matrix(self, X):
        probas = self._collect_probas(X)
        predicted_labels = np.argmax(probas, axis=2)
        predicted_labels[predicted_labels == 0] = -1
        values = np.max(probas, axis=2)
        return (predicted_labels * values).T

    def _collect_probas(self, X):
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_generator.estimators_])

    def _restricted_master_problem(self, y_kernel_matrix):
        raise NotImplementedError("Restricted master problem not implemented.")

    def _initialize_alphas(self, n_examples):
        raise NotImplementedError("Alpha weights initialization function is not implemented.")

    def evaluate_metrics(self, X, y, metrics_list=None, functions_list=None):
        if metrics_list is None:
            metrics_list = [zero_one_loss, zero_one_loss_per_example]

        if functions_list is None:
            functions_list = []

        # Predict, evaluate metrics.
        classification_matrix = self._binary_classification_matrix(X)
        predictions = sign(classification_matrix.dot(self.weights_))

        if self.save_iteration_as_hyperparameter_each is None:
            metrics_results = {}
            for metric in metrics_list:
                metrics_results[metric.__name__] = metric(y, predictions)

            metrics_dataframe = ResultsDataFrame([metrics_results])
            return metrics_dataframe

        # If we collected iteration informations to add a hyperparameter, we add an index with the hyperparameter name
        # and return a ResultsDataFrame containing one row per hyperparameter value.
        metrics_dataframe = ResultsDataFrame()
        for t, weights in iteritems(self.collected_weight_vectors_):
            predictions = sign(classification_matrix[:, :t + 1].dot(weights))
            metrics_results = {metric.__name__: metric(y, predictions) for metric in metrics_list}
            for function in functions_list:
                metrics_results[function.__name__] = function(classification_matrix[:, :t + 1], y, weights)

            # We add other collected information.
            metrics_results['chosen_columns'] = self.chosen_columns_[t]
            metrics_results['dual_constraint_violation'] = self.collected_dual_constraint_violations_[t]

            metrics_dataframe = metrics_dataframe.append(ResultsDataFrame([metrics_results], index=[t]))

        metrics_dataframe.index.name = 'hp__n_iterations'
        return metrics_dataframe

class CqBoostClassifier(ColumnGenerationClassifier):
    def __init__(self, mu=0.001, epsilon=1e-08, n_max_iterations=None, estimators_generator=None, save_iteration_as_hyperparameter_each=None):
        super(CqBoostClassifier, self).__init__(epsilon, n_max_iterations, estimators_generator, dual_constraint_rhs=0,
                                                save_iteration_as_hyperparameter_each=save_iteration_as_hyperparameter_each)
        # TODO: Vérifier la valeur de nu (dual_constraint_rhs) à l'initialisation, mais de toute manière ignorée car
        # on ne peut pas quitter la boucle principale avec seulement un votant.
        self.mu = mu

    def _restricted_master_problem(self, y_kernel_matrix, previous_w=None, previous_alpha=None):
        n_examples, n_hypotheses = y_kernel_matrix.shape

        m_eye = np.eye(n_examples)
        m_ones = np.ones((n_examples, 1))

        qp_a = np.vstack((np.hstack((-y_kernel_matrix, m_eye)),
                          np.hstack((np.ones((1, n_hypotheses)), np.zeros((1, n_examples))))))

        qp_b = np.vstack((np.zeros((n_examples, 1)),
                          np.array([1.0]).reshape((1, 1))))

        qp_g = np.vstack((np.hstack((-np.eye(n_hypotheses), np.zeros((n_hypotheses, n_examples)))),
                          np.hstack((np.zeros((1, n_hypotheses)), - 1.0 / n_examples * m_ones.T))))

        qp_h = np.vstack((np.zeros((n_hypotheses, 1)),
                          np.array([-self.mu]).reshape((1, 1))))

        qp = ConvexProgram()
        qp.quadratic_func = 2.0 / n_examples * np.vstack((np.hstack((np.zeros((n_hypotheses, n_hypotheses)), np.zeros((n_hypotheses, n_examples)))),
                                                        np.hstack((np.zeros((n_examples, n_hypotheses)), m_eye))))

        qp.add_equality_constraints(qp_a, qp_b)
        qp.add_inequality_constraints(qp_g, qp_h)

        if previous_w is not None:
            qp.initial_values = np.append(previous_w, [0])

        try:
            solver_result = qp.solve(abstol=1e-10, reltol=1e-10, feastol=1e-10, return_all_information=True)
            w = np.asarray(np.array(solver_result['x']).T[0])[:n_hypotheses]

            # The alphas are the Lagrange multipliers associated with the equality constraints (returned as the y vector in CVXOPT).
            dual_variables = np.asarray(np.array(solver_result['y']).T[0])
            alpha = dual_variables[:n_examples]

            # Set the dual constraint right-hand side to be equal to the last lagrange multiplier (nu).
            # Hack: do not change nu if the QP didn't fully solve...
            if solver_result['dual slack'] <= 1e-8:
                self.dual_constraint_rhs = dual_variables[-1]
                # logging.info('Updating dual constraint rhs: {}'.format(self.dual_constraint_rhs))

        except:
            logging.warning('QP Solving failed at iteration {}.'.format(n_hypotheses))
            if previous_w is not None:
                w = np.append(previous_w, [0])
            else:
                w = np.array([1.0 / n_hypotheses] * n_hypotheses)

            if previous_alpha is not None:
                alpha = previous_alpha
            else:
                alpha = self._initialize_alphas(n_examples)

        return w, alpha

    def _initialize_alphas(self, n_examples):
        return 1.0 / n_examples * np.ones((n_examples,))


class CQBoost(CqBoostClassifier):

    def __init__(self, random_state, **kwargs):
        super(CQBoost, self).__init__(
            mu=kwargs['mu'],
            epsilon=kwargs['epsilon'],
            n_max_iterations= kwargs['n_max_iterations'],
            )

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return False

    def paramsToSrt(self, nIter=1):
        """Used for weighted linear early fusion to generate random search sets"""
        paramsSet = []
        for _ in range(nIter):
            paramsSet.append({"mu": 0.001,
                              "epsilon": 1e-08,
                              "n_max_iterations": None})
        return paramsSet

    def getKWARGS(self, args):
        """Used to format kwargs for the parsed args"""
        kwargsDict = {}
        kwargsDict['mu'] = 0.001
        kwargsDict['epsilon'] = 1e-08
        kwargsDict['n_max_iterations'] = None
        return kwargsDict

    def genPipeline(self):
        return Pipeline([('classifier', CqBoostClassifier())])

    def genParamsDict(self, randomState):
        return {"classifier__mu": [0.001],
                "classifier__epsilon": [1e-08],
                "classifier__n_max_iterations": [None]}

    def genBestParams(self, detector):
        return {"mu": detector.best_params_["classifier__mu"],
                "epsilon": detector.best_params_["classifier__epsilon"],
                "n_max_iterations": detector.best_params_["classifier__n_max_iterations"]}

    def genParamsFromDetector(self, detector):
        nIter = len(detector.cv_results_['param_classifier__mu'])
        return [("mu", np.array([0.001 for _ in range(nIter)])),
                ("epsilon", np.array(detector.cv_results_['param_classifier__epsilon'])),
                ("n_max_iterations", np.array(detector.cv_results_['param_classifier__n_max_iterations']))]

    def getConfig(self, config):
        if type(config) is not dict:  # Used in late fusion when config is a classifier
            return "\n\t\t- CQBoost with mu : " + str(config.mu) + ", epsilon : " + str(
                config.epsilon + ", n_max_iterations : " + str(config.n_max_iterations))
        else:
            return "\n\t\t- CQBoost with mu : " + str(config["mu"]) + ", epsilon : " + str(
                   config["epsilon"] + ", n_max_iterations : " + str(config["n_max_iterations"]))


    def getInterpret(self, classifier, directory):
        interpretString = ""
        return interpretString






def canProbas():
    return False


def fit(DATASET, CLASS_LABELS, randomState, NB_CORES=1, **kwargs):
    """Used to fit the monoview classifier with the args stored in kwargs"""
    classifier = CqBoostClassifier(mu=kwargs['mu'],
                                   epsilon=kwargs['epsilon'],
                                   n_max_iterations=kwargs["n_max_iterations"],)
                                   # random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"mu": 10**-randomState.uniform(0.5, 1.5),
                          "epsilon": 10**-randomState.randint(1, 15),
                          "n_max_iterations": None})
    return paramsSet


def getKWARGS(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {}
    kwargsDict['mu'] = args.CQB_mu
    kwargsDict['epsilon'] = args.CQB_epsilon
    kwargsDict['n_max_iterations'] = None
    return kwargsDict


def genPipeline():
    return Pipeline([('classifier', CqBoostClassifier())])


def genParamsDict(randomState):
    return {"classifier__mu": CustomUniform(loc=.5, state=2, multiplier='e-'),
                "classifier__epsilon": CustomRandint(low=1, high=15, multiplier='e-'),
                "classifier__n_max_iterations": [None]}


def genBestParams(detector):
    return {"mu": detector.best_params_["classifier__mu"],
                "epsilon": detector.best_params_["classifier__epsilon"],
                "n_max_iterations": detector.best_params_["classifier__n_max_iterations"]}


def genParamsFromDetector(detector):
    nIter = len(detector.cv_results_['param_classifier__mu'])
    return [("mu", detector.cv_results_['param_classifier__mu']),
            ("epsilon", np.array(detector.cv_results_['param_classifier__epsilon'])),
            ("n_max_iterations", np.array(detector.cv_results_['param_classifier__n_max_iterations']))]


def getConfig(config):
    if type(config) is not dict:  # Used in late fusion when config is a classifier
        return "\n\t\t- CQBoost with mu : " + str(config.mu) + ", epsilon : " + str(
            config.epsilon) + ", n_max_iterations : " + str(config.n_max_iterations)
    else:
        return "\n\t\t- CQBoost with mu : " + str(config["mu"]) + ", epsilon : " + str(
            config["epsilon"]) + ", n_max_iterations : " + str(config["n_max_iterations"])


def getInterpret(classifier, directory):
    dotted = False
    interpretString = "\t CQBoost permformed classification with weights : \n"
    interpretString += np.array2string(classifier.weights_, precision=4, separator=',', suppress_small=True)
    interpretString += "\n \t It used {} iterations to converge".format(len(classifier.weights_))
    if len(classifier.weights_) == classifier.n_max_iterations:
        interpretString += ", and used all available iterations, "
    else:
        dotted = True
        interpretString += "."
    if len(classifier.weights_) == classifier.n_total_hypotheses_:
        interpretString += ", and all the voters have been used."
    elif not dotted:
        interpretString += "."
    interpretString += "\n\t Selected voters : \n"
    interpretString += str(classifier.chosen_columns_)
    interpretString += "\n\t and they voted : \n"
    interpretString += np.array2string(classifier.classification_matrix[:, classifier.chosen_columns_], precision=4, separator=',', suppress_small=True)
    np.savetxt(directory+"voters.csv", classifier.classification_matrix[:, classifier.chosen_columns_], delimiter=',')
    np.savetxt(directory + "weights.csv", classifier.weights_, delimiter=',')
    return interpretString





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
        assert(n_lines == n_columns)

        if self._linear_func is not None:
            assert(np.shape(quad_matrix)[0] == self._n_variables)
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
                assert(np.shape(lin_vector)[0] == self._n_variables)

            else:
                self._n_variables = np.shape(lin_vector)[0]

            self._linear_func = lin_vector

    def add_inequality_constraints(self, inequality_matrix, inequality_values):
        if inequality_matrix is None:
            logging.info("Empty inequality constraint: ignoring!")
            return

        self._assert_objective_function_is_set()

        if 1 in np.shape(inequality_matrix) or len(np.shape(inequality_matrix)) == 1:
            inequality_matrix = _as_line_matrix(inequality_matrix)
        else:
            inequality_matrix = _as_matrix(inequality_matrix)

        inequality_values = _as_column_matrix(inequality_values)
        assert np.shape(inequality_matrix)[1] == self._n_variables
        assert np.shape(inequality_values)[1] == 1

        if self._inequality_constraints_matrix is None:
            self._inequality_constraints_matrix = inequality_matrix
        else:
            self._inequality_constraints_matrix = np.append(self._inequality_constraints_matrix,
                                                            inequality_matrix, axis=0)

        if self._inequality_constraints_values is None:
            self._inequality_constraints_values = inequality_values
        else:
            self._inequality_constraints_values = np.append(self._inequality_constraints_values,
                                                            inequality_values, axis=0)

    def add_equality_constraints(self, equality_matrix, equality_values):
        if equality_matrix is None:
            logging.info("Empty equality constraint: ignoring!")
            return

        self._assert_objective_function_is_set()

        if 1 in np.shape(equality_matrix) or len(np.shape(equality_matrix)) == 1:
            equality_matrix = _as_line_matrix(equality_matrix)
        else:
            equality_matrix = _as_matrix(equality_matrix)

        equality_values = _as_matrix(equality_values)
        assert np.shape(equality_matrix)[1] == self._n_variables
        assert np.shape(equality_values)[1] == 1

        if self._equality_constraints_matrix is None:
            self._equality_constraints_matrix = equality_matrix
        else:
            self._equality_constraints_matrix = np.append(self._equality_constraints_matrix,
                                                          equality_matrix, axis=0)

        if self._equality_constraints_values is None:
            self._equality_constraints_values = equality_values
        else:
            self._equality_constraints_values = np.append(self._equality_constraints_values,
                                                          equality_values, axis=0)

    def add_lower_bound(self, lower_bound):
        if lower_bound is not None:
            self._assert_objective_function_is_set()
            self._lower_bound_values = np.array([lower_bound] * self._n_variables)

    def add_upper_bound(self, upper_bound):
        if upper_bound is not None:
            self._assert_objective_function_is_set()
            self._upper_bound_values = np.array([upper_bound] * self._n_variables)

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
            self._inequality_constraints_matrix = cvxopt_matrix(self._inequality_constraints_matrix)

        if self._inequality_constraints_values is not None:
            self._inequality_constraints_values = cvxopt_matrix(self._inequality_constraints_values)

        if self._equality_constraints_matrix is not None:
            self._equality_constraints_matrix = cvxopt_matrix(self._equality_constraints_matrix)

        if self._equality_constraints_values is not None:
            self._equality_constraints_values = cvxopt_matrix(self._equality_constraints_values)

    def _assert_objective_function_is_set(self):
        assert self._n_variables is not None

    def solve(self, solver="cvxopt", feastol=1e-7, abstol=1e-7, reltol=1e-6, return_all_information=False):

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
                    ret = qp(self.quadratic_func, self.linear_func, self._inequality_constraints_matrix,
                             self._inequality_constraints_values, self._equality_constraints_matrix,
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
                                               np.asarray(self.linear_func.T).reshape(self.n_variables,).tolist()))

                if self._inequality_constraints_matrix is not None:
                    inequality_linear = []
                    for line in self._inequality_constraints_matrix:
                        inequality_linear.append([names, line.tolist()[0]])
                    p.linear_constraints.add(lin_expr=inequality_linear,
                                             rhs=np.asarray(self._inequality_constraints_values.T).tolist()[0],
                                             senses="L"*len(self._inequality_constraints_values))

                if self._equality_constraints_matrix is not None:
                    equality_linear = []
                    for line in self._equality_constraints_matrix:
                        equality_linear.append([names, line.tolist()[0]])
                    p.linear_constraints.add(lin_expr=equality_linear,
                                             rhs=np.asarray(self._equality_constraints_values.T).tolist()[0],
                                             senses="E"*len(self._equality_constraints_values))

                if self._lower_bound_values is not None:
                    p.variables.set_lower_bounds(zip(names, self._lower_bound_values))

                if self._upper_bound_values is not None:
                    p.variables.set_upper_bounds(zip(names, self._upper_bound_values))

                p.solve()

                logging.info("Solution status = {} : {}".format(p.solution.get_status(),
                                                                p.solution.status[p.solution.get_status()]))
                logging.info("Solution value  = {}".format(p.solution.get_objective_value()))

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
                    model.constrain(self._inequality_constraints_matrix * q <= self._inequality_constraints_values)
                if self._equality_constraints_matrix is not None:
                    model.constrain(self._equality_constraints_matrix * q == self._equality_constraints_values)
                if self._lower_bound_values is not None:
                    model.constrain(q >= self._lower_bound_values)
                if self._upper_bound_values is not None:
                    model.constrain(q <= self._upper_bound_values)

                value = model.minimize(0.5 * q.T * self._quadratic_func * q + self.linear_func.T * q)

                logging.info("Solution value  = {}".format(value))

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

        assert len(self.classes_) == 2, "DecisionStumpsVoter only supports binary classification"
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
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

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
        positive_class = np.argwhere(X[:, self.attribute_index] > self.threshold)
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
    def __init__(self, n_stumps_per_attribute=10, self_complemented=False):
        super(StumpsClassifiersGenerator, self).__init__(self_complemented)
        self.n_stumps_per_attribute = n_stumps_per_attribute

    def fit(self, X, y):
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
        ranges = (maximums - minimums) / (self.n_stumps_per_attribute + 1)

        self.estimators_ = [DecisionStumpClassifier(i, minimums[i] + ranges[i] * stump_number, 1).fit(X, y)
                            for i in range(X.shape[1]) for stump_number in range(1, self.n_stumps_per_attribute + 1)
                            if ranges[i] != 0]

        if self.self_complemented:
            self.estimators_ += [DecisionStumpClassifier(i, minimums[i] + ranges[i] * stump_number, -1).fit(X, y)
                                 for i in range(X.shape[1]) for stump_number in range(1, self.n_stumps_per_attribute + 1)
                                 if ranges[i] != 0]

        self.estimators_ = np.asarray(self.estimators_)
        return self

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


def zero_one_loss(y_target, y_estimate, confidences=1):
    if len(y_target) == 0:
        return 0.0
    return np.mean(y_target != y_estimate)


def zero_one_loss_per_example(y_target, y_estimate, confidences=1):
    if len(y_target) == 0:
        return 0.0
    return (y_target != y_estimate).astype(np.int)


class ResultsDataFrame(pd.DataFrame):
    """A ResultsDataFrame is a DataFrame with the following information:

    - A 'dataset' column that contains the dataset name
    - Hyperparamer columns, named 'hp__HPNAME', where HPNAME is the name of the hyperparameter
    - Columns containing informations about that depend on the dataset and hyperparameters, for example the risk.

    """
    @property
    def datasets_list(self):
        """Returns the sorted list of datasets.

        """
        return sorted(set(self['dataset']))

    @property
    def hyperparameters_list(self):
        """Returns a sorted list of hyperparameter names, without the 'hp__' prefix.

        """
        return sorted(column.split('hp__')[1] for column in self.columns if column.startswith('hp__'))

    @property
    def hyperparameters_list_with_prefix(self):
        return sorted(column for column in self.columns if column.startswith('hp__'))

    @property
    def metrics_list(self):
        return sorted(column for column in self.columns if not column.startswith('hp__') and column != 'dataset')

    @property
    def hyperparameters_with_values(self):
        """Returns a dictionary that contains the hyperparameter names (without the 'hp__' prefix), and
        associated values that are present in the DataFrame.

        """
        hyperparameters = [column for column in self.columns if column.startswith('hp__')]

        hyperparameters_dict = {}
        tmp_dict = self[hyperparameters].to_dict()

        for key, value in iteritems(tmp_dict):
            hyperparameters_dict[key.split('hp__')[1]] = list(value.values())[0] if len(value) == 1 else sorted(set(value.values()))

        return hyperparameters_dict

    @property
    def hyperparameters_with_values_per_dataset(self):
        """Returns a dictionary of dictionaries that contains for each dataset, the hyperparameter names (without the
        'hp__' prefix), and associated values that are present in the DataFrame.

        """
        hyperparameters = [column for column in self.columns if column.startswith('hp__')]

        hyperparameters_dict = {}
        for dataset in self.datasets_list:
            tmp_dict = self[self.dataset == dataset][hyperparameters].to_dict()
            hyperparameters_dict[dataset] = {}

            for key, value in iteritems(tmp_dict):
                hyperparameters_dict[dataset][key.split('hp__')[1]] = list(value.values())[0] if len(value) == 1 else sorted(value.values())

        return hyperparameters_dict

    def results_optimizing_metric(self, metric_to_optimize='cv_mean__valid__zero_one_loss', minimize=True, tie_breaking_functions_ordered_dict=None):
        function = min if minimize else max

        # We extract all the rows that have the best value for the metric to optimize.
        optimal_results = self[self.groupby('dataset', sort=False)[metric_to_optimize].transform(function) == self[metric_to_optimize]]

        # We tie the breaks by applying the tie breaking functions (in the order of the dictionary). If hyperparameters are missing, we simply
        # use the median for each hyperparameter, in a fixed (reproduceable) order.
        if tie_breaking_functions_ordered_dict is None:
            tie_breaking_functions_ordered_dict = OrderedDict()
        else:
            # Avoid side effects and ensures that the dictionary is an OrderedDict before we add missing hyperparameters.
            tie_breaking_functions_ordered_dict = OrderedDict(tie_breaking_functions_ordered_dict.copy())

        for hyperparameter in sorted(self.hyperparameters_list):
            if hyperparameter not in tie_breaking_functions_ordered_dict.keys():
                tie_breaking_functions_ordered_dict[hyperparameter] = np.median

        for hyperparameter, tie_breaking_function in iteritems(tie_breaking_functions_ordered_dict):
            optimal_results = optimal_results[optimal_results.groupby('dataset')['hp__' + hyperparameter].transform(partial(get_optimal_value_in_list, tie_breaking_function)) == optimal_results['hp__' + hyperparameter]]

        return ResultsDataFrame(optimal_results)

    def get_dataframe_with_metrics_as_one_column(self, metrics_to_keep=None):
        new_dataframe = ResultsDataFrame()

        if metrics_to_keep is None:
            metrics_to_keep = self.metrics_list

        for metric in metrics_to_keep:
            columns = self.hyperparameters_list_with_prefix + [metric]
            if 'dataset' in self:
                columns.append('dataset')

            tmp = self.loc[:, columns]
            tmp.columns = [c if c != metric else 'value' for c in tmp.columns]
            tmp.loc[:, 'metric'] = metric
            new_dataframe = new_dataframe.append(tmp, ignore_index=True)

        return new_dataframe


def get_optimal_value_in_list(optimum_function, values_list):
    """Given a list of values and an optimal value, returns the value from the list that is the closest to the optimum,
    given by optimum_function applied to the same list.

    >>> get_optimal_value_in_list(np.median, [2, 4, 5, 6])
    4

    """
    values_list = sorted(list(values_list))
    return values_list[np.argmin(np.array([scipy.spatial.distance.euclidean(value, optimum_function(values_list)) for value in values_list]))]

import logging
import math
import time
from collections import defaultdict

import numpy as np
import numpy.ma as ma
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from .BoostUtils import StumpsClassifiersGenerator, ConvexProgram, sign, \
    BaseBoost, TreeClassifiersGenerator
from ... import metrics


class ColumnGenerationClassifier(BaseEstimator, ClassifierMixin, BaseBoost):
    def __init__(self, mu=0.01, epsilon=1e-06, n_max_iterations=100,
                 estimators_generator="Stumps", dual_constraint_rhs=0, max_depth=1,
                 save_iteration_as_hyperparameter_each=None, random_state=None):
        super(ColumnGenerationClassifier, self).__init__()
        self.epsilon = epsilon
        self.n_max_iterations = n_max_iterations
        self.estimators_generator = estimators_generator
        self.dual_constraint_rhs = dual_constraint_rhs
        self.mu = mu
        self.max_depth=max_depth
        self.train_time = 0
        self.plotted_metric = metrics.zero_one_loss
        self.random_state = random_state

    def fit(self, X, y):
        if scipy.sparse.issparse(X):
            X = np.array(X.todense())

        y[y == 0] = -1

        if self.estimators_generator is "Stumps":
            self.estimators_generator = StumpsClassifiersGenerator(
                n_stumps_per_attribute=self.n_stumps, self_complemented=True)
        elif self.estimators_generator is "Trees":
            self.estimators_generator = TreeClassifiersGenerator(
                max_depth=self.max_depth, n_trees=self.n_stumps,
                self_complemented=True)

        self.estimators_generator.fit(X, y)
        self.classification_matrix = self._binary_classification_matrix(X)
        self.c_bounds = []

        self.infos_per_iteration_ = defaultdict(list)

        m, n = self.classification_matrix.shape
        self.chosen_columns_ = []
        self.n_total_hypotheses_ = n
        self.n_total_examples = m
        self.train_shape = self.classification_matrix.shape

        y_kernel_matrix = np.multiply(y.reshape((len(y), 1)),
                                      self.classification_matrix)

        # Initialization
        alpha = self._initialize_alphas(m)
        self.initialize()
        self.train_metrics = []
        self.gammas = []
        self.list_weights = []
        self.bounds = []
        self.previous_votes = []
        # w = [0.5,0.5]
        w = None
        self.collected_weight_vectors_ = {}
        self.collected_dual_constraint_violations_ = {}
        start = time.time()

        for k in range(min(n,
                           self.n_max_iterations if self.n_max_iterations is not None else np.inf)):
            # Find worst weak hypothesis given alpha.
            h_values = ma.array(
                np.squeeze(np.array((alpha).T.dot(y_kernel_matrix).T)),
                fill_value=-np.inf)

            if self.chosen_columns_:
                h_values[self.chosen_columns_] = ma.masked

            worst_h_index = ma.argmax(h_values)

            # Check for optimal solution. We ensure at least one complete iteration is done as the initialization
            # values might provide a degenerate initial solution.
            if self.chosen_columns_:
                if h_values[
                    worst_h_index] <= self.dual_constraint_rhs + self.epsilon and len(
                        self.chosen_columns_) > 0:
                    break

            # Append the weak hypothesis.
            self.chosen_columns_.append(worst_h_index)
            self.matrix_to_optimize = self.get_matrix_to_optimize(
                y_kernel_matrix, w)

            # Solve restricted master for new costs.
            w, alpha = self._restricted_master_problem(previous_w=w,
                                                       previous_alpha=alpha)
            cbound = self.compute_empiric_cbound(w, y_kernel_matrix)
            self.c_bounds.append(cbound)
            self.list_weights.append(w)

            self.update_values(h_values, worst_h_index, alpha, w)

            margins = self.get_margins(w)
            signs_array = np.array([int(x) for x in sign(margins)])
            self.train_metrics.append(self.plotted_metric.score(y, signs_array))
            self.gammas.append(accuracy_score(y, signs_array))
            self.bounds.append(
                math.exp(-2 * np.sum(np.square(np.array(self.gammas)))))

        self.nb_opposed_voters = self.check_opposed_voters()
        self.compute_weights_(w)
        # self.weights_ = w
        self.estimators_generator.choose(self.chosen_columns_)
        end = time.time()

        self.train_time = end - start
        y[y == -1] = 0
        return self

    def predict(self, X):
        start = time.time()
        check_is_fitted(self, 'weights_')

        if scipy.sparse.issparse(X):
            logging.warning('Converting sparse matrix to dense matrix.')
            X = np.array(X.todense())

        classification_matrix = self._binary_classification_matrix(X)
        margins = np.squeeze(
            np.asarray(np.dot(classification_matrix, self.weights_)))

        signs_array = np.array([int(x) for x in sign(margins)])
        signs_array[signs_array == -1] = 0
        end = time.time()
        self.predict_time = end - start
        self.step_predict(classification_matrix)
        return signs_array

    def compute_empiric_cbound(self, w, y_kernel_matrix):
        cbound = 1 - (1.0 / self.n_total_examples) * (np.sum(
            np.average(y_kernel_matrix[:, self.chosen_columns_], axis=1,
                       weights=w)) ** 2 /
                                                      np.sum(np.average(
                                                          y_kernel_matrix[:,
                                                          self.chosen_columns_],
                                                          axis=1,
                                                          weights=w) ** 2))
        return cbound

    def step_predict(self, classification_matrix):
        if classification_matrix.shape != self.train_shape:
            self.step_decisions = np.zeros(classification_matrix.shape)
            self.step_prod = np.zeros(classification_matrix.shape)
            for weight_index in range(self.weights_.shape[0] - 1):
                margins = np.sum(classification_matrix[:, :weight_index + 1] *
                                 self.list_weights[weight_index], axis=1)
                signs_array = np.array([int(x) for x in sign(margins)])
                signs_array[signs_array == -1] = 0
                self.step_decisions[:, weight_index] = signs_array
                self.step_prod[:, weight_index] = np.sum(
                    classification_matrix[:, :weight_index + 1] * self.weights_[
                                                                  :weight_index + 1],
                    axis=1)

    def initialize(self):
        pass

    def update_values(self, h_values=None, worst_h_index=None, alpha=None,
                      w=None):
        pass

    def get_margins(self, w):
        margins = np.squeeze(np.asarray(
            np.dot(self.classification_matrix[:, self.chosen_columns_], w)))
        return margins

    def compute_weights_(self, w=None):
        self.weights_ = w

    def get_matrix_to_optimize(self, y_kernel_matrix, w=None):
        return y_kernel_matrix[:, self.chosen_columns_]

    # def _binary_classification_matrix(self, X):
    #     probas = self._collect_probas(X)
    #     predicted_labels = np.argmax(probas, axis=2)
    #     predicted_labels[predicted_labels == 0] = -1
    #     values = np.max(probas, axis=2)
    #     return (predicted_labels * values).T
    #
    # def _collect_probas(self, X):
    #     return np.asarray([clf.predict_proba(X) for clf in self.estimators_generator.estimators_])

    def _restricted_master_problem(self, previous_w=None, previous_alpha=None):
        n_examples, n_hypotheses = self.matrix_to_optimize.shape

        m_eye = np.eye(n_examples)
        m_ones = np.ones((n_examples, 1))

        qp_a = np.vstack((np.hstack((-self.matrix_to_optimize, m_eye)),
                          np.hstack((np.ones((1, n_hypotheses)),
                                     np.zeros((1, n_examples))))))

        qp_b = np.vstack((np.zeros((n_examples, 1)),
                          np.array([1.0]).reshape((1, 1))))

        qp_g = np.vstack((np.hstack(
            (-np.eye(n_hypotheses), np.zeros((n_hypotheses, n_examples)))),
                          np.hstack((np.zeros((1, n_hypotheses)),
                                     - 1.0 / n_examples * m_ones.T))))

        qp_h = np.vstack((np.zeros((n_hypotheses, 1)),
                          np.array([-self.mu]).reshape((1, 1))))

        qp = ConvexProgram()
        qp.quadratic_func = 2.0 / n_examples * np.vstack((np.hstack((np.zeros(
            (n_hypotheses, n_hypotheses)), np.zeros(
            (n_hypotheses, n_examples)))),
                                                          np.hstack((np.zeros((
                                                                              n_examples,
                                                                              n_hypotheses)),
                                                                     m_eye))))

        qp.add_equality_constraints(qp_a, qp_b)
        qp.add_inequality_constraints(qp_g, qp_h)

        if previous_w is not None:
            qp.initial_values = np.append(previous_w, [0])

        try:
            solver_result = qp.solve(abstol=1e-10, reltol=1e-10, feastol=1e-10,
                                     return_all_information=True)
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
            logging.warning(
                'QP Solving failed at iteration {}.'.format(n_hypotheses))
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

# class CqBoostClassifier(ColumnGenerationClassifier):
#     def __init__(self, mu=0.001, epsilon=1e-08, n_max_iterations=None, estimators_generator=None, save_iteration_as_hyperparameter_each=None):
#         super(CqBoostClassifier, self).__init__(epsilon, n_max_iterations, estimators_generator, dual_constraint_rhs=0,
#                                                 save_iteration_as_hyperparameter_each=save_iteration_as_hyperparameter_each)
#         # TODO: Verifier la valeur de nu (dual_constraint_rhs) a l'initialisation, mais de toute maniere ignoree car
#         # on ne peut pas quitter la boucle principale avec seulement un votant.
#         self.mu = mu
#         self.train_time = 0
#
#     def _restricted_master_problem(self, y_kernel_matrix, previous_w=None, previous_alpha=None):
#         n_examples, n_hypotheses = y_kernel_matrix.shape
#
#         m_eye = np.eye(n_examples)
#         m_ones = np.ones((n_examples, 1))
#
#         qp_a = np.vstack((np.hstack((-y_kernel_matrix, m_eye)),
#                           np.hstack((np.ones((1, n_hypotheses)), np.zeros((1, n_examples))))))
#
#         qp_b = np.vstack((np.zeros((n_examples, 1)),
#                           np.array([1.0]).reshape((1, 1))))
#
#         qp_g = np.vstack((np.hstack((-np.eye(n_hypotheses), np.zeros((n_hypotheses, n_examples)))),
#                           np.hstack((np.zeros((1, n_hypotheses)), - 1.0 / n_examples * m_ones.T))))
#
#         qp_h = np.vstack((np.zeros((n_hypotheses, 1)),
#                           np.array([-self.mu]).reshape((1, 1))))
#
#         qp = ConvexProgram()
#         qp.quadratic_func = 2.0 / n_examples * np.vstack((np.hstack((np.zeros((n_hypotheses, n_hypotheses)), np.zeros((n_hypotheses, n_examples)))),
#                                                         np.hstack((np.zeros((n_examples, n_hypotheses)), m_eye))))
#
#         qp.add_equality_constraints(qp_a, qp_b)
#         qp.add_inequality_constraints(qp_g, qp_h)
#
#         if previous_w is not None:
#             qp.initial_values = np.append(previous_w, [0])
#
#         try:
#             solver_result = qp.solve(abstol=1e-10, reltol=1e-10, feastol=1e-10, return_all_information=True)
#             w = np.asarray(np.array(solver_result['x']).T[0])[:n_hypotheses]
#
#             # The alphas are the Lagrange multipliers associated with the equality constraints (returned as the y vector in CVXOPT).
#             dual_variables = np.asarray(np.array(solver_result['y']).T[0])
#             alpha = dual_variables[:n_examples]
#
#             # Set the dual constraint right-hand side to be equal to the last lagrange multiplier (nu).
#             # Hack: do not change nu if the QP didn't fully solve...
#             if solver_result['dual slack'] <= 1e-8:
#                 self.dual_constraint_rhs = dual_variables[-1]
#                 # logging.info('Updating dual constraint rhs: {}'.format(self.dual_constraint_rhs))
#
#         except:
#             logging.warning('QP Solving failed at iteration {}.'.format(n_hypotheses))
#             if previous_w is not None:
#                 w = np.append(previous_w, [0])
#             else:
#                 w = np.array([1.0 / n_hypotheses] * n_hypotheses)
#
#             if previous_alpha is not None:
#                 alpha = previous_alpha
#             else:
#                 alpha = self._initialize_alphas(n_examples)
#
#         return w, alpha
#
#     def _initialize_alphas(self, n_examples):
#         return 1.0 / n_examples * np.ones((n_examples,))

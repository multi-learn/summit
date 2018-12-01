import scipy
import logging
import numpy as np
import numpy.ma as ma
from collections import defaultdict
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import time

from ..Monoview.MonoviewUtils import CustomUniform, CustomRandint, BaseMonoviewClassifier
from ..Monoview.Additions.BoostUtils import StumpsClassifiersGenerator, sign, getInterpretBase, BaseBoost


class ColumnGenerationClassifierv21(BaseEstimator, ClassifierMixin, BaseBoost):
    def __init__(self, epsilon=1e-06, n_max_iterations=None, estimators_generator=None, dual_constraint_rhs=0, save_iteration_as_hyperparameter_each=None, random_state=42):
        super(ColumnGenerationClassifierv21, self).__init__()
        self.epsilon = epsilon
        self.n_max_iterations = n_max_iterations
        self.estimators_generator = estimators_generator
        self.dual_constraint_rhs = dual_constraint_rhs
        self.save_iteration_as_hyperparameter_each = save_iteration_as_hyperparameter_each
        self.random_state = random_state

    def fit(self, X, y):
        if scipy.sparse.issparse(X):
            logging.info('Converting to dense matrix.')
            X = np.array(X.todense())

        if self.estimators_generator is None:
            self.estimators_generator = StumpsClassifiersGenerator(n_stumps_per_attribute=self.n_stumps, self_complemented=True)

        y[y == 0] = -1

        self.estimators_generator.fit(X, y)
        self.classification_matrix = self._binary_classification_matrix(X)


        self.weights_ = []
        self.infos_per_iteration_ = defaultdict(list)

        m, n = self.classification_matrix.shape
        y_kernel_matrix = np.multiply(y.reshape((len(y), 1)), self.classification_matrix)

       # Initialization

        w = None
        self.collected_weight_vectors_ = {}
        self.collected_dual_constraint_violations_ = {}

        example_weights = self._initialize_alphas(m).reshape((m,1))

        self.chosen_columns_ = []
        self.fobidden_columns = []
        self.edge_scores = []
        self.example_weights_ = [example_weights]
        self.train_accuracies = []
        self.previous_votes = []

        self.n_total_hypotheses_ = n
        self.n_total_examples = m
        # print("\n \t\t Start fit\n")
        for k in range(min(n, self.n_max_iterations if self.n_max_iterations is not None else np.inf)):
            # Find worst weak hypothesis given alpha.
            new_voter_index, criterion = self._find_new_voter(example_weights, y_kernel_matrix, "pseudo_h")

            # Append the weak hypothesis.
            self.chosen_columns_.append(new_voter_index)
            self.fobidden_columns.append(new_voter_index)
            new_voter_margin = y_kernel_matrix[:, self.chosen_columns_[-1]].reshape((m, 1))
            self.edge_scores.append(criterion)

            if w is None:
                self.previous_vote = new_voter_margin
                w = 1
                self.weights_.append(w)
                example_weights = self._update_example_weights(example_weights, y_kernel_matrix, m)
                self.example_weights_.append(example_weights)
                self.train_accuracies.append(accuracy_score(y, np.sign(self.previous_vote)))
                continue

            # ---- On resoud le probleme a deux votants analytiquement.
            w = self._solve_two_weights_min_c(new_voter_margin, example_weights)
            if w[0] == "break":
                self.chosen_columns_.pop()
                self.break_cause = w[1]
                break
            self.previous_vote = np.matmul(np.concatenate((self.previous_vote, new_voter_margin), axis=1),
                                           w).reshape((m,1))

            # We collect iteration information for later evaluation.
            self.weights_.append(w[-1])

            self.weights = np.array(self.weights_)
            self.final_vote_weights = np.array([np.prod(1 - self.weights[t + 1:]) * self.weights[t] if t <
                                                                                                         self.weights.shape[
                                                                                                             0] - 1 else
                                                self.weights[t] for t in range(self.weights.shape[0])])
            margins = np.squeeze(np.asarray(np.matmul(self.classification_matrix[:, self.chosen_columns_],
                                                      self.final_vote_weights)))
            signs_array = np.array([int(x) for x in sign(margins)])
            self.train_accuracies.append(accuracy_score(y, signs_array))

            # ---- On change l'edge
            example_weights = self._update_example_weights(example_weights, y_kernel_matrix, m)
            self.example_weights_.append(example_weights)

        self.nb_opposed_voters = self.check_opposed_voters()
        self.estimators_generator.estimators_ = self.estimators_generator.estimators_[self.chosen_columns_]

        y[y == -1] = 0

        return self

    def predict(self, X):
        start = time.time()
        check_is_fitted(self, 'weights_')

        if scipy.sparse.issparse(X):
            logging.warning('Converting sparse matrix to dense matrix.')
            X = np.array(X.todense())
        classification_matrix = self._binary_classification_matrix(X)
        self.weights_ = np.array(self.weights_)
        self.final_vote_weights = np.array([np.prod(1-self.weights_[t+1:])*self.weights_[t] if t < self.weights_.shape[0]-1 else self.weights_[t] for t in range(self.weights_.shape[0]) ])
        margins = np.squeeze(np.asarray(np.matmul(classification_matrix, self.final_vote_weights)))
        signs_array = np.array([int(x) for x in sign(margins)])
        signs_array[signs_array == -1 ] = 0
        end = time.time()
        self.predict_time = end-start
        return signs_array

    def _find_new_voter(self, example_weights, y_kernel_matrix, type="pseudo_h"):
        if type == "pseudo_h":
            pseudo_h_values = ma.array(np.squeeze(np.array(example_weights.T.dot(y_kernel_matrix).T)), fill_value=-np.inf)
            pseudo_h_values[self.fobidden_columns] = ma.masked
            worst_h_index = ma.argmax(pseudo_h_values)
            return worst_h_index, pseudo_h_values[worst_h_index]
        elif type == "random":
            new_index = self.random_state.choice(np.arange(self.n_total_hypotheses_))
            while new_index in self.fobidden_columns:
                new_index = self.random_state.choice(np.arange(self.n_total_hypotheses_))
            return new_index, 100

    def _update_example_weights(self, example_weights, y_kernel_matrix, m):
        if len(self.weights_)==1:
            example_weights[self.previous_vote == -1] *= 2
            example_weights[self.previous_vote == 1 ] /= 2
            pass
        else:
            weights = np.array(self.weights_)
            current_vote_weights = np.array([np.prod(1 - weights[t + 1:]) * weights[t] if t <
                                                                                          weights.shape[
                                                                                                          0] - 1 else
                                             weights[t] for t in range(weights.shape[0])]).reshape((weights.shape[0], 1))
            weighted_margin = np.matmul(y_kernel_matrix[:, self.chosen_columns_], current_vote_weights)
            example_weights = np.multiply(example_weights,
                                          np.exp((1 - np.sum(weighted_margin, axis=1) /
                                                      np.sum(weighted_margin, axis=1))).reshape((m, 1)))
        return example_weights

    def _solve_two_weights_min_c(self, next_column, example_weights):
        m = next_column.shape[0]
        zero_diag = np.ones((m, m)) - np.identity(m)

        weighted_previous_vote = self.previous_vote.reshape((m, 1))
        weighted_next_column = next_column.reshape((m,1))

        mat_prev = np.repeat(weighted_previous_vote, m, axis=1) * zero_diag
        mat_next = np.repeat(weighted_next_column, m, axis=1) * zero_diag

        self.B2 = np.sum((weighted_previous_vote - weighted_next_column) ** 2)
        self.B1 = np.sum(2 * weighted_next_column * (weighted_previous_vote - 2 * weighted_next_column * weighted_next_column))
        self.B0 = np.sum(weighted_next_column * weighted_next_column)

        self.A2 = self.B2 + np.sum((mat_prev - mat_next) * np.transpose(mat_prev - mat_next))
        self.A1 = self.B1 + np.sum(mat_prev * np.transpose(mat_next) - mat_next * np.transpose(mat_prev) - 2 * mat_next * np.transpose(mat_next))
        self.A0 = self.B0 + np.sum(mat_next * np.transpose(mat_next))

        C2 = (self.A1 * self.B2 - self.A2 * self.B1)
        C1 = 2 * (self.A0 * self.B2 - self.A2 * self.B0)
        C0 = self.A0 * self.B1 - self.A1 * self.B0

        if C2 == 0:
            if C1 == 0:
                return np.array([0.5, 0.5])
            elif abs(C1) > 0:
                return np.array([0., 1.])
            else:
                return ['break', "the derivate was constant."]
        elif C2 == 0:
            return ["break", "the derivate was affine."]

        sols = np.roots(np.array([C2, C1, C0]))

        is_acceptable, sol = self._analyze_solutions(sols)
        if is_acceptable:
            # print("cb", self._cborn(sol))
            return np.array([sol, 1-sol])
        else:
            return ["break", sol]

    def _analyze_solutions(self, sols):
        if sols.shape[0] == 1:
            if self._cborn(sols[0]) < self._cborn(sols[0]+1):
                best_sol = sols[0]
            else:
                return False, " the only solution was a maximum."
        elif sols.shape[0] == 2:
            best_sol = self._best_sol(sols)
        else:
            return False, " no solution were found."

        if 0 < best_sol < 1:
            return True, self._best_sol(sols)

        elif best_sol <= 0:
            return False, " the minimum was below 0."
        else:
            return False, " the minimum was over 1."

    def _cborn(self, sol):
        return 1 - (self.A2*sol**2 + self.A1*sol + self.A0)/(self.B2*sol**2 + self.B1*sol + self.B0)

    def _best_sol(self, sols):
        values = np.array([self._cborn(sol) for sol in sols])
        return sols[np.argmin(values)]

    def _restricted_master_problem(self, y_kernel_matrix):
        raise NotImplementedError("Restricted master problem not implemented.")


class CqBoostClassifierv21(ColumnGenerationClassifierv21):
    def __init__(self, mu=0.001, epsilon=1e-08, n_max_iterations=None, estimators_generator=None, save_iteration_as_hyperparameter_each=None, random_state=42):
        super(CqBoostClassifierv21, self).__init__(epsilon, n_max_iterations, estimators_generator, dual_constraint_rhs=0,
                                                   save_iteration_as_hyperparameter_each=save_iteration_as_hyperparameter_each, random_state=random_state)
        self.train_time = 0
        self.mu = mu

    def _initialize_alphas(self, n_examples):
        return 1.0 / n_examples * np.ones((n_examples,))


class CQBoostv21(CqBoostClassifierv21, BaseMonoviewClassifier):

    def __init__(self, random_state=None, mu=0.01, epsilon=1e-06, **kwargs):
        super(CQBoostv21, self).__init__(
            random_state=random_state,
            mu=mu,
            epsilon=epsilon
        )
        self.param_names = ["mu", "epsilon"]
        self.distribs = [CustomUniform(loc=0.5, state=1.0, multiplier="e-"),
                         CustomRandint(low=1, high=15, multiplier="e-")]
        self.classed_params = []
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory):
        return getInterpretBase(self, directory, "CQBoostv21", self.weights_, self.break_cause)

    def get_name_for_fusion(self):
        return "CQ21"


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"mu": args.CQB2_mu,
                  "epsilon": args.CQB2_epsilon}
    return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"mu": 10**-randomState.uniform(0.5, 1.5),
                          "epsilon": 10**-randomState.randint(1, 15)})
    return paramsSet
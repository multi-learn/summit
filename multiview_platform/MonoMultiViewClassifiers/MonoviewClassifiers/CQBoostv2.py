import scipy
import logging
import numpy.ma as ma
from collections import defaultdict
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import time

from ..Monoview.MonoviewUtils import CustomRandint, CustomUniform
from ..Monoview.Additions.BoostUtils import StumpsClassifiersGenerator, ConvexProgram, sign, getInterpretBase, BaseBoost


class ColumnGenerationClassifierv2(BaseEstimator, ClassifierMixin, BaseBoost):
    def __init__(self, epsilon=1e-06, n_max_iterations=None, estimators_generator=None, dual_constraint_rhs=0, save_iteration_as_hyperparameter_each=None):
        super(ColumnGenerationClassifierv2, self).__init__()
        self.epsilon = epsilon
        self.n_max_iterations = n_max_iterations
        self.estimators_generator = estimators_generator
        self.dual_constraint_rhs = dual_constraint_rhs
        self.save_iteration_as_hyperparameter_each = save_iteration_as_hyperparameter_each

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

        alpha = self._initialize_alphas(m)

        self.chosen_columns_ = []
        self.edge_scores = []
        self.alphas = []
        self.train_accuracies = []
        self.previous_votes = []

        self.n_total_hypotheses_ = n
        self.n_total_examples = m

        for k in range(min(n, self.n_max_iterations if self.n_max_iterations is not None else np.inf)):
            # Find worst weak hypothesis given alpha.
            h_values = ma.array(np.squeeze(np.array((alpha).T.dot(y_kernel_matrix).T)), fill_value=-np.inf)
            h_values[self.chosen_columns_] = ma.masked
            worst_h_index = ma.argmax(h_values)

            if h_values[worst_h_index] <= self.dual_constraint_rhs + self.epsilon and len(self.chosen_columns_) > 0:
                break

            # Append the weak hypothesis.
            self.chosen_columns_.append(worst_h_index)
            self.edge_scores.append(h_values[worst_h_index])

            if w is not None:
                self.matrix_to_optimize = np.concatenate((np.matmul(self.matrix_to_optimize, w).reshape((m, 1)),
                                                          y_kernel_matrix[:, self.chosen_columns_[-1]].reshape((m, 1))),
                                                         axis=1)
            else:
                self.matrix_to_optimize = y_kernel_matrix[:, self.chosen_columns_[-1]].reshape((m, 1))

            # Solve restricted master for new costs.
            w, alpha = self._restricted_master_problem(self.matrix_to_optimize, previous_w=w, previous_alpha=alpha)

            self.alphas.append(alpha)


            # We collect iteration information for later evaluation.
            self.weights_.append(w[-1])

            self.weights = np.array(self.weights_)
            self.final_vote_weights = np.array([np.prod(1 - self.weights[t + 1:]) * self.weights_[t] if t <
                                                                                                         self.weights.shape[
                                                                                                             0] - 1 else
                                                self.weights[t] for t in range(self.weights.shape[0])])
            margins = np.squeeze(np.asarray(np.matmul(self.classification_matrix[:,self.chosen_columns_],
                                                      self.final_vote_weights)))
            signs_array = np.array([int(x) for x in sign(margins)])
            self.train_accuracies.append(accuracy_score(y, signs_array))

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
        signs_array[signs_array == -1] = 0
        end = time.time()
        self.predict_time = end-start
        return signs_array

    def _restricted_master_problem(self, y_kernel_matrix):
        raise NotImplementedError("Restricted master problem not implemented.")



class CqBoostClassifierv2(ColumnGenerationClassifierv2):
    def __init__(self, mu=0.001, epsilon=1e-08, n_max_iterations=None, estimators_generator=None, save_iteration_as_hyperparameter_each=None):
        super(CqBoostClassifierv2, self).__init__(epsilon, n_max_iterations, estimators_generator, dual_constraint_rhs=0,
                                                save_iteration_as_hyperparameter_each=save_iteration_as_hyperparameter_each)
        # TODO: Vérifier la valeur de nu (dual_constraint_rhs) à l'initialisation, mais de toute manière ignorée car
        # on ne peut pas quitter la boucle principale avec seulement un votant.
        self.mu = mu
        self.train_time = 0

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


class CQBoostv2(CqBoostClassifierv2):

    def __init__(self, random_state, **kwargs):
        super(CQBoostv2, self).__init__(
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
        return Pipeline([('classifier', CqBoostClassifierv2())])

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
    start = time.time()
    classifier = CqBoostClassifierv2(mu=kwargs['mu'],
                                   epsilon=kwargs['epsilon'],
                                   n_max_iterations=kwargs["n_max_iterations"],)
                                   # random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    end = time.time()
    classifier.train_time =end-start
    return classifier


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"mu": randomState.uniform(1e-02, 10**(-0.5)),
                          "epsilon": 10**-randomState.randint(1, 15),
                          "n_max_iterations": None})
    return paramsSet


def getKWARGS(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {}
    kwargsDict['mu'] = args.CQB2_mu
    kwargsDict['epsilon'] = args.CQB2_epsilon
    kwargsDict['n_max_iterations'] = None
    return kwargsDict


def genPipeline():
    return Pipeline([('classifier', CqBoostClassifierv2())])


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
    return [("mu", np.array([0.001 for _ in range(nIter)])),
            ("epsilon", np.array(detector.cv_results_['param_classifier__epsilon'])),
            ("n_max_iterations", np.array(detector.cv_results_['param_classifier__n_max_iterations']))]


def getConfig(config):
    if type(config) is not dict:  # Used in late fusion when config is a classifier
        return "\n\t\t- CQBoostv2 with mu : " + str(config.mu) + ", epsilon : " + str(
            config.epsilon) + ", n_max_iterations : " + str(config.n_max_iterations)
    else:
        return "\n\t\t- CQBoostv2 with mu : " + str(config["mu"]) + ", epsilon : " + str(
            config["epsilon"]) + ", n_max_iterations : " + str(config["n_max_iterations"])


def getInterpret(classifier, directory):
    return getInterpretBase(classifier, directory, "CQBoostv2", classifier.final_vote_weights)


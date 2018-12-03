import scipy
import logging
import numpy as np
import numpy.ma as ma
from collections import defaultdict
import math
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
import time
import matplotlib.pyplot as plt

from .BoostUtils import StumpsClassifiersGenerator, sign, BaseBoost, getInterpretBase, get_accuracy_graph
from ... import Metrics


class ColumnGenerationClassifierQar(BaseEstimator, ClassifierMixin, BaseBoost):
    def __init__(self, n_max_iterations=None, estimators_generator=None,
                 random_state=42, self_complemented=True, twice_the_same=False,
                 c_bound_choice = True, random_start = True,
                 n_stumps_per_attribute=None, use_r=True,
                 plotted_metric=Metrics.zero_one_loss):
        super(ColumnGenerationClassifierQar, self).__init__()

        if type(random_state) is int:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
        self.train_time = 0
        self.n_max_iterations = n_max_iterations
        self.estimators_generator = estimators_generator
        self.self_complemented = self_complemented
        self.twice_the_same = twice_the_same
        self.c_bound_choice = c_bound_choice
        self.random_start = random_start
        self.plotted_metric = plotted_metric
        if n_stumps_per_attribute:
            self.n_stumps = n_stumps_per_attribute
        self.use_r = use_r
        self.printed_args_name_list = ["n_max_iterations", "self_complemented", "twice_the_same",
                                       "c_bound_choice", "random_start",
                                       "n_stumps", "use_r"]

    def set_params(self, **params):
        self.self_complemented = params["self_complemented"]
        self.twice_the_same = params["twice_the_same"]
        self.c_bound_choice = params["c_bound_choice"]
        self.random_start = params["random_start"]

    def fit(self, X, y):
        start = time.time()

        formatted_X, formatted_y = self.format_X_y(X, y)

        self.init_info_containers()

        m,n,y_kernel_matrix = self.init_hypotheses(formatted_X, formatted_y)

        self.n_total_hypotheses_ = n
        self.n_total_examples = m

        self.init_boosting(m, formatted_y, y_kernel_matrix)
        self.break_cause = " the maximum number of iterations was attained."

        for k in range(min(n-1, self.n_max_iterations-1 if self.n_max_iterations is not None else np.inf)):

            # Print dynamically the step and the error of the current classifier
            print("Resp. bound : {}, {}/{}, eps :{}".format(self.respected_bound, k+2, self.n_max_iterations, self.voter_perfs[-1]), end="\r")

            sol, new_voter_index = self.choose_new_voter(y_kernel_matrix, formatted_y)

            if type(sol) == str:
                self.break_cause = new_voter_index  #
                break

            self.append_new_voter(new_voter_index)

            voter_perf = self.compute_voter_perf(formatted_y)

            # if epsilon == 0. or math.log((1 - epsilon) / epsilon) == math.inf:
            #     self.chosen_columns_.pop()
            #     self.break_cause = " epsilon was too small."
            #     break

            self.compute_voter_weight(voter_perf)

            self.update_example_weights(formatted_y)

            self.update_info_containers(formatted_y, voter_perf, k)


        self.nb_opposed_voters = self.check_opposed_voters()
        self.estimators_generator.estimators_ = self.estimators_generator.estimators_[self.chosen_columns_]

        self.weights_ = np.array(self.weights_)
        self.weights_/= np.sum(self.weights_)

        formatted_y[formatted_y == -1] = 0
        formatted_y = formatted_y.reshape((m,))

        end = time.time()
        self.train_time = end - start
        return self

    def predict(self, X):
        start = time.time()
        check_is_fitted(self, 'weights_')
        if scipy.sparse.issparse(X):
            logging.warning('Converting sparse matrix to dense matrix.')
            X = np.array(X.todense())
        classification_matrix = self._binary_classification_matrix(X)
        margins = np.squeeze(np.asarray(np.matmul(classification_matrix, self.weights_)))
        signs_array = np.array([int(x) for x in sign(margins)])
        signs_array[signs_array == -1] = 0
        end = time.time()
        self.predict_time = end - start
        return signs_array

    def update_info_containers(self, y, voter_perf, k):
        """Is used at each iteration to compute and store all the needed quantities for later analysis"""
        self.example_weights_.append(self.example_weights)
        self.previous_vote = np.matmul(
            self.classification_matrix[:, self.chosen_columns_],
            np.array(self.weights_).reshape((k + 2, 1))).reshape((self.n_total_examples, 1))
        self.previous_votes.append(self.previous_vote)

        self.previous_margins.append(
            np.multiply(y, self.previous_vote))
        train_metric = self.plotted_metric.score(y, np.sign(self.previous_vote))
        if self.use_r:
            bound = self.bounds[-1] * math.sqrt(1 - voter_perf ** 2)
        else:
            bound = np.prod(np.sqrt(1-4*np.square(0.5-np.array(self.voter_perfs))))

        if train_metric > bound:
            self.respected_bound = False

        self.train_metrics.append(train_metric)
        self.bounds.append(bound)

    def compute_voter_weight(self, voter_perf):
        """used to compute the voter's weight according to the specified method (edge or error) """
        if self.use_r:
            self.q = 0.5 * math.log((1 + voter_perf) / (1 - voter_perf))
        else:
            self.q = math.log((1 - voter_perf) / voter_perf)
        self.weights_.append(self.q)

    def compute_voter_perf(self, formatted_y):
        """Used to computer the performance (error or edge) of the selected voter"""
        if self.use_r:
            r = self._compute_r(formatted_y)
            self.voter_perfs.append(r)
            return r
        else:
            epsilon = self._compute_epsilon(formatted_y)
            self.voter_perfs.append(epsilon)
            return epsilon

    def append_new_voter(self, new_voter_index):
        """Used to append the voter to the majority vote"""
        self.chosen_columns_.append(new_voter_index)
        self.new_voter = self.classification_matrix[:, new_voter_index].reshape(
            (self.n_total_examples, 1))

    def choose_new_voter(self, y_kernel_matrix, formatted_y):
        """Used to chhoose the voter according to the specified criterion (margin or C-Bound"""
        if self.c_bound_choice:
            sol, new_voter_index = self._find_new_voter(y_kernel_matrix,
                                                        formatted_y)
        else:
            new_voter_index, sol = self._find_best_weighted_margin(
                y_kernel_matrix)
        return sol, new_voter_index


    def init_boosting(self, m, y, y_kernel_matrix):
        """THis initialization corressponds to the first round of boosting with equal weights for each examples and the voter chosen by it's margin."""
        self.example_weights = self._initialize_alphas(m).reshape((m, 1))

        self.previous_margins.append(np.multiply(y, y))
        self.example_weights_.append(self.example_weights)
        if self.random_start:
            first_voter_index = self.random_state.choice(
                self.get_possible(y_kernel_matrix, y))
        else:
            first_voter_index, _ = self._find_best_weighted_margin(
                y_kernel_matrix)

        self.chosen_columns_.append(first_voter_index)
        self.new_voter = self.classification_matrix[:,
                         first_voter_index].reshape((m, 1))

        self.previous_vote = self.new_voter

        if self.use_r:
            r = self._compute_r(y)
            self.voter_perfs.append(r)
        else:
            epsilon = self._compute_epsilon(y)
            self.voter_perfs.append(epsilon)


        if self.use_r:
            self.q = 0.5 * math.log((1 + r) / (1 - r))
        else:
            self.q = math.log((1 - epsilon) / epsilon)
        self.weights_.append(self.q)

        # Update the distribution on the examples.
        self.update_example_weights(y)
        self.example_weights_.append(self.example_weights)

        self.previous_margins.append(
            np.multiply(y, self.previous_vote))

        train_metric =self.plotted_metric.score(y, np.sign(self.previous_vote))
        if self.use_r:
            bound = math.sqrt(1 - r ** 2)
        else:
            bound = np.prod(np.sqrt(1-4*np.square(0.5-np.array(epsilon))))

        if train_metric > bound:
            self.respected_bound = False

        self.train_metrics.append(train_metric)

        self.bounds.append(bound)

    def format_X_y(self, X, y):
        """Formats the data  : X -the examples- and y -the labels- to be used properly by the algorithm """
        if scipy.sparse.issparse(X):
            logging.info('Converting to dense matrix.')
            X = np.array(X.todense())
        # Initialization
        y[y == 0] = -1
        y = y.reshape((y.shape[0], 1))
        return X, y

    def init_hypotheses(self, X, y):
        """Inintialization for the hyptotheses used to build the boosted vote"""
        if self.estimators_generator is None:
            self.estimators_generator = StumpsClassifiersGenerator(n_stumps_per_attribute=self.n_stumps,
                                                                   self_complemented=self.self_complemented)
        self.estimators_generator.fit(X, y)
        self.classification_matrix = self._binary_classification_matrix(X)

        m, n = self.classification_matrix.shape
        y_kernel_matrix = np.multiply(y, self.classification_matrix)
        return m,n,y_kernel_matrix

    def init_info_containers(self):
        """Initialize the containers that will be collected at each iteration for the analysis"""
        self.weights_ = []
        self.chosen_columns_ = []
        self.fobidden_columns = []
        self.c_bounds = []
        self.voter_perfs = []
        self.example_weights_ = []
        self.train_metrics = []
        self.bounds = []
        self.previous_votes = []
        self.previous_margins = []
        self.respected_bound = True

    def _compute_epsilon(self,y):
        """Updating the error variable, the old fashioned way uses the whole majority vote to update the error"""
        ones_matrix = np.zeros(y.shape)
        ones_matrix[np.multiply(y, self.new_voter.reshape(y.shape)) < 0] = 1  # can np.divide if needed
        epsilon = np.average(ones_matrix, weights=self.example_weights, axis=0)
        return epsilon

    def _compute_r(self, y):
        ones_matrix = np.ones(y.shape)
        ones_matrix[np.multiply(y, self.new_voter.reshape(y.shape)) < 0] = -1  # can np.divide if needed
        r = np.average(ones_matrix, weights=self.example_weights, axis=0)
        return r

    def update_example_weights(self, y):
        """Old fashioned exaple weights update uses the whole majority vote, the other way uses only the last voter."""
        new_weights = self.example_weights.reshape((self.n_total_examples, 1))*np.exp(-self.q*np.multiply(y,self.new_voter))
        self.example_weights = new_weights/np.sum(new_weights)

    def _find_best_weighted_margin(self, y_kernel_matrix, upper_bound=1.0, lower_bound=0.0):
        """Finds the new voter by choosing the one that has the best weighted margin between 0.5 and 0.55
        to avoid too god voters that will get all the votes weights"""
        weighted_kernel_matrix = np.multiply(y_kernel_matrix, self.example_weights.reshape((self.n_total_examples, 1)))
        pseudo_h_values = ma.array(np.sum(weighted_kernel_matrix, axis=0), fill_value=-np.inf)
        pseudo_h_values[self.chosen_columns_] = ma.masked
        acceptable_indices = np.where(np.logical_and(np.greater(upper_bound, pseudo_h_values), np.greater(pseudo_h_values, lower_bound)))[0]
        if acceptable_indices.size > 0:
            worst_h_index = self.random_state.choice(acceptable_indices)
            return worst_h_index, [0]
        else:
            return " no margin over random and acceptable", ""

    def _is_not_too_wrong(self, hypothese, y):
        """Check if the weighted margin is better than random"""
        ones_matrix = np.zeros(y.shape)
        ones_matrix[hypothese.reshape(y.shape) < 0] = 1
        epsilon = np.average(ones_matrix, weights=self.example_weights, axis=0)
        return epsilon < 0.5

    def get_possible(self, y_kernel_matrix, y):
        """Get all the indices of the hypothesis that are good enough to be chosen"""
        possibleIndices = []
        for hypIndex, hypothese in enumerate(np.transpose(y_kernel_matrix)):
            if self._is_not_too_wrong(hypothese, y):
                possibleIndices.append(hypIndex)
        return np.array(possibleIndices)

    def _find_new_voter(self, y_kernel_matrix, y):
        """Here, we solve the two_voters_mincq_problem for each potential new voter,
        and select the one that has the smallest minimum"""
        c_borns = []
        possible_sols = []
        indices = []
        causes = []
        for hypothese_index, hypothese in enumerate(y_kernel_matrix.transpose()):
            if (hypothese_index not in self.chosen_columns_ or self.twice_the_same) \
                    and set(self.chosen_columns_)!={hypothese_index} \
                    and self._is_not_too_wrong(hypothese, y):
                w = self._solve_one_weight_min_c(hypothese, y)
                if w[0] != "break":
                    c_borns.append(self._cbound(w[0]))
                    possible_sols.append(w)
                    indices.append(hypothese_index)
                else:
                    causes.append(w[1])
        if not causes:
            causes = ["no feature was better than random and acceptable"]
        if c_borns:
            min_c_born_index = ma.argmin(c_borns)
            self.c_bounds.append(c_borns[min_c_born_index])
            selected_sol = possible_sols[min_c_born_index]
            selected_voter_index = indices[min_c_born_index]
            return selected_sol, selected_voter_index
        else:
            return "break", " and ".join(set(causes))

    def _solve_one_weight_min_c(self, next_column, y):
        """Here we solve the min C-bound problem for two voters using one weight only and return the best weight
        No precalc because longer ; see the "derivee" latex document for more precision"""
        m = next_column.shape[0]
        zero_diag = np.ones((m, m)) - np.identity(m)
        weighted_previous_sum = np.multiply(y, self.previous_vote.reshape((m, 1)))
        weighted_next_column = np.multiply(next_column.reshape((m,1)), self.example_weights.reshape((m,1)))

        self.B2 = np.sum(weighted_next_column ** 2)
        self.B1 = np.sum(2 * weighted_next_column * weighted_previous_sum)
        self.B0 = np.sum(weighted_previous_sum ** 2)

        M2 = np.sum(np.multiply(np.matmul(weighted_next_column, np.transpose(weighted_next_column)), zero_diag))
        M1 = np.sum(np.multiply(np.matmul(weighted_previous_sum, np.transpose(weighted_next_column)) +
                                np.matmul(weighted_next_column, np.transpose(weighted_previous_sum))
                                , zero_diag))
        M0 = np.sum(np.multiply(np.matmul(weighted_previous_sum, np.transpose(weighted_previous_sum)), zero_diag))

        self.A2 = self.B2 + M2
        self.A1 = self.B1 + M1
        self.A0 = self.B0 + M0

        C2 = (M1 * self.B2 - M2 * self.B1)
        C1 = 2 * (M0 * self.B2 - M2 * self.B0)
        C0 = M0 * self.B1 - M1 * self.B0
        if C2 == 0:
            if C1 == 0:
                return ['break', "the derivate was constant"]
            else :
                is_acceptable, sol = self._analyze_solutions_one_weight(np.array(float(C0)/C1).reshape((1,1)))
                if is_acceptable:
                    return np.array([sol])
        try:
            sols = np.roots(np.array([C2, C1, C0]))
        except:
            return ["break", "nan"]

        is_acceptable, sol = self._analyze_solutions_one_weight(sols)
        if is_acceptable:
            return np.array([sol])
        else:
            return ["break", sol]

    def _analyze_solutions_one_weight(self, sols):
        """"We just check that the solution found by np.roots is acceptable under our constraints
        (real, a minimum and over 0)"""
        if sols.shape[0] == 1:
            if self._cbound(sols[0]) < self._cbound(sols[0] + 1):
                best_sol = sols[0]
            else:
                return False, "the only solution was a maximum."
        elif sols.shape[0] == 2:
            best_sol = self._best_sol(sols)
        else:
            return False, "no solution were found"

        if isinstance(best_sol, complex):
            return False, "the sol was complex"
        else:
            return True, best_sol

    def _cbound(self, sol):
        """Computing the objective function"""
        return 1 - (self.A2*sol**2 + self.A1*sol + self.A0)/(self.B2*sol**2 + self.B1*sol + self.B0)/self.n_total_examples

    def _best_sol(self, sols):
        """Return the best min in the two possible sols"""
        values = np.array([self._cbound(sol) for sol in sols])
        return sols[np.argmin(values)]

    def _initialize_alphas(self, n_examples):
        """Initialize the examples wieghts"""
        return 1.0 / n_examples * np.ones((n_examples,))

    def getInterpretQar(self, directory):
        """Used to interpret the functionning of the algorithm"""
        path = "/".join(directory.split("/")[:-1])
        try:
            import os
            os.makedirs(path+"/gif_images")
        except:
            raise
        filenames=[]
        max_weight = max([np.max(examples_weights) for examples_weights in self.example_weights_])
        min_weight = min([np.max(examples_weights) for examples_weights in self.example_weights_])
        for iterIndex, examples_weights in enumerate(self.example_weights_):
            r = np.array(examples_weights)
            theta = np.arange(self.n_total_examples)
            colors = np.sign(self.previous_margins[iterIndex])
            fig = plt.figure(figsize=(5, 5), dpi=80)
            ax = fig.add_subplot(111)
            c = ax.scatter(theta, r, c=colors, cmap='RdYlGn', alpha=0.75)
            ax.set_ylim(min_weight, max_weight)
            filename = path+"/gif_images/"+str(iterIndex)+".png"
            filenames.append(filename)
            plt.savefig(filename)
            plt.close()

        import imageio
        images = []
        logging.getLogger("PIL").setLevel(logging.WARNING)
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(path+'/weights.gif', images, duration=1. / 2)
        import shutil
        shutil.rmtree(path+"/gif_images")
        get_accuracy_graph(self.voter_perfs, self.__class__.__name__, directory + 'voter_perfs.png', "Errors")
        interpretString = getInterpretBase(self, directory, "QarBoost", self.weights_, self.break_cause)

        args_dict = dict((arg_name, str(self.__dict__[arg_name])) for arg_name in self.printed_args_name_list)
        interpretString += "\n \n With arguments : \n"+u'\u2022 '+ ("\n"+u'\u2022 ').join(['%s: \t%s' % (key, value)
                                                                                           for (key, value) in args_dict.items()])
        if not self.respected_bound:
            interpretString += "\n\n The bound was not respected"

        return interpretString






from abc import ABC, abstractmethod
import numpy as np


class BlackBoxFunction(ABC):

    @abstractmethod
    def simulation(self, scenario):
        pass


class SimpleBlackBoxFunction(BlackBoxFunction):

    def __init__(self, black_box_function_name, return_max, sampler, normalization_quantile_level, N_empirical_distribution, verbose):
        self._black_box_function_name                = black_box_function_name
        self._return_max                             = return_max
        self._sampler                                = sampler
        self._normalization_quantile_level           = normalization_quantile_level
        self._N_empirical_distribution               = N_empirical_distribution
        self._verbose                                = verbose
        self._unormalized_max_empirical_distribution = []
        self.normalized_max_empirical_distribution   = []
        self._normalization_quantile                 = None

    def _unormalized_black_box_function(self, scenario):
        if self._black_box_function_name == "Linear black-box function":
            return np.array([np.sum(scenario)])
        if self._black_box_function_name == "Quadratic black-box function":
            return np.array([np.sum(scenario**2)])
        if self._black_box_function_name == "Exponential black-box function":
            return np.array([np.sum(np.exp(scenario))])
        if self._black_box_function_name == "2D quadratic black-box function":
            return np.array([np.sum(scenario**2), np.sum(scenario)**2])

    def _compute_unormalized_max_empirical_distribution(self):
        for i in range(self._N_empirical_distribution):
            if i % int(self._N_empirical_distribution / 100) == 0 and self._verbose:
                print("Unormalized max empirical distribution computation : " + str(int(100 * i / self._N_empirical_distribution)) + "%")
            scenario = self._sampler.draw_properly_scaled_scenario()
            max_y    = np.max(self._unormalized_black_box_function(scenario))
            self._unormalized_max_empirical_distribution.append(max_y)
        self._unormalized_max_empirical_distribution = np.array(self._unormalized_max_empirical_distribution)

    def _compute_normalized_max_empirical_distribution(self):
        self.normalized_max_empirical_distribution = self._unormalized_max_empirical_distribution / self._normalization_quantile

    def compute_normalization_quantile(self):
        self._compute_unormalized_max_empirical_distribution()
        self._normalization_quantile = np.quantile(self._unormalized_max_empirical_distribution, self._normalization_quantile_level)
        self._compute_normalized_max_empirical_distribution()

    def simulation(self, scenario):
        if self._return_max:
            return np.array([np.max(self._unormalized_black_box_function(scenario) / self._normalization_quantile)])
        else:
            return self._unormalized_black_box_function(scenario) / self._normalization_quantile

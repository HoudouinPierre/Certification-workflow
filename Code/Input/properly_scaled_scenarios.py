from abc import ABC, abstractmethod
import numpy as np


class ProperlyScaledScenario(ABC):

    @abstractmethod
    def properly_scaled(self, scenario):
        pass


class SimpleProperlyScaledScenario(ProperlyScaledScenario):

    def __init__(self, properly_scaled_scenarios_max_L2_norm):
        self._properly_scaled_scenarios_max_L2_norm = properly_scaled_scenarios_max_L2_norm

    def properly_scaled(self, scenario):
        return np.sum(scenario**2) < self._properly_scaled_scenarios_max_L2_norm

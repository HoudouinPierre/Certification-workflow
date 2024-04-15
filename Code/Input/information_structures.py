from abc import ABC, abstractmethod
import numpy as np


class InformationStructure(ABC):

    @abstractmethod
    def draw_scenario(self):
        pass


class SimpleInformationStructure(InformationStructure):

    def __init__(self, dimension, information_structure_bound):
        self._dimension                      = dimension
        self._information_structure_bound    = information_structure_bound

    def draw_scenario(self):
        return np.random.rand(self._dimension) * self._information_structure_bound

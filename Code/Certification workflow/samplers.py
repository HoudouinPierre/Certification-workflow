from abc import ABC, abstractmethod


class Sampler(ABC):

    @abstractmethod
    def draw_scenario(self):
        pass

    @abstractmethod
    def draw_properly_scaled_scenario(self):
        pass


class SimpleSampler(Sampler):

    def __init__(self, information_structure, properly_scaled_scenario):
        self._information_structure    = information_structure
        self._properly_scaled_scenario = properly_scaled_scenario

    def draw_scenario(self):
        scenario = self._information_structure.draw_scenario()
        return scenario

    def draw_properly_scaled_scenario(self):
        properly_scaled_scenario, scenario = False, None
        while not properly_scaled_scenario:
            scenario                 = self._information_structure   .draw_scenario()
            properly_scaled_scenario = self._properly_scaled_scenario.properly_scaled(scenario)
        return scenario

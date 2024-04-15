import numpy as np


class SimpleProxyWorkflow:

    def __init__(self, sampler, black_box_function, proxy_use_tool, proxy, Omega, N_workflow, simulate_all_scenarios_for_workflow_evaluation, verbose):
        self._sampler                                        = sampler
        self._black_box_function                             = black_box_function
        self._proxy_use_tool                                 = proxy_use_tool
        self._proxy                                          = proxy
        self._Omega                                          = Omega
        self._N_workflow                                     = N_workflow
        self._simulate_all_scenarios_for_workflow_evaluation = simulate_all_scenarios_for_workflow_evaluation
        self._verbose                                        = verbose

    def _proxy_use(self, proxy_congestion_probability):
        return self._proxy_use_tool(proxy_congestion_probability)

    def _draw_properly_scaled_scenario(self):
        return self._sampler.draw_properly_scaled_scenario()

    def _simulation(self, scenario):
        return self._black_box_function.simulation(scenario)

    def _proxy_prediction(self, X, Y, scenario):
        proxy_congestion_probability, posterior_mean, posterior_std, hyperparameters = self._proxy.simulation(X, Y, scenario)
        return proxy_congestion_probability, posterior_mean, posterior_std, hyperparameters

    def certification(self):
        X, Y, outcomes = [], [], []
        for iter1 in range(self._N_workflow):
            if iter1 % int(self._N_workflow / 100) == 0 and self._verbose:
                print("Certification - " + str(int(100 * iter1 / self._N_workflow)) + "%")
            scenario                                                                     = self._draw_properly_scaled_scenario()
            proxy_congestion_probability, posterior_mean, posterior_std, hyperparameters = self._proxy_prediction(X, Y, scenario)
            proxy_can_be_used                                                            = self._proxy_use(proxy_congestion_probability)
            if proxy_can_be_used:
                y                      = self._simulation(scenario) if self._simulate_all_scenarios_for_workflow_evaluation else None
                outcome                = {"Scenario"               : scenario                    , "Simulation output"            : y                           ,
                                          "Posterior mean"         : posterior_mean              , "Posterior std"                : posterior_std               ,
                                          "Congestion probability" : proxy_congestion_probability, "Proxy congestion probability" : proxy_congestion_probability,
                                          "Hyperparameters"        : hyperparameters             , "Computation method"           : "Proxy"}
            else:
                y                      = self._simulation(scenario)
                congestion_probability = 1 if np.max(y) > 1 else 0
                outcome                = {"Scenario"               : scenario                    , "Simulation output"            : y                           ,
                                          "Posterior mean"         : posterior_mean              , "Posterior std"                : posterior_std               ,
                                          "Congestion probability" : congestion_probability      , "Proxy congestion probability" : proxy_congestion_probability,
                                          "Hyperparameters"        : hyperparameters             , "Computation method"           : "Black-box function"}
                X.append(scenario)
                Y.append(y       )
            outcomes.append(outcome)
        for iter2 in range(self._N_workflow):
            if iter2 % int(self._N_workflow / 100) == 0 and self._verbose:
                print("GP result recomputation - " + str(int(100 * iter1 / self._N_workflow)) + "%")
            if outcomes[iter2]["Computation method"] == "Proxy":
                scenario                                                                     = outcomes[iter2]["Scenario"]
                y                                                                            = outcomes[iter2]["Simulation output"]
                proxy_congestion_probability, posterior_mean, posterior_std, hyperparameters = self._proxy_prediction(X, Y, scenario)
                outcome                = {"Scenario"               : scenario                    , "Simulation output"            : y                           ,
                                          "Posterior mean"         : posterior_mean              , "Posterior std"                : posterior_std               ,
                                          "Congestion probability" : proxy_congestion_probability, "Proxy congestion probability" : proxy_congestion_probability,
                                          "Hyperparameters"        : hyperparameters             , "Computation method"           : "Proxy"}
                outcomes[iter2] = outcome
        return outcomes

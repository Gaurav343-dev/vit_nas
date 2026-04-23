from abc import ABC, abstractmethod

import torch

from utils.measurements import get_macs, get_peak_memory

class BaseSearch(ABC):
    def __init__(self, efficiency_predictor, accuracy_predictor):
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor

    @abstractmethod
    def run_search(self, constraint, **kwargs):
        raise NotImplementedError("Search method must be implemented by subclasses")

    @abstractmethod
    def random_valid_sample(self, constraint):
        raise NotImplementedError(
            "Random valid sampling method must be implemented by subclasses"
        )

class AnalyticalEfficiencyPredictor:
    def __init__(self, net, img_size: int):
        self.net = net
        self.img_size = img_size

    def get_efficiency(self, model):
        # model has already had set_active_subnet() called on it before this
        # data_shape = (1, 3, self.img_size, self.img_size)
        macs = get_macs(model)
        # TODO: implement get_peak_memory to get actual peak memory usage during a forward pass.
        peak_memory_kb = get_peak_memory(model, img_size=self.img_size, batch_size=1, unit="KB", method="analytical")
        ################ YOUR CODE ENDS HERE ################

        return dict(millionMACs=macs / 1e6, KBPeakMemory=peak_memory_kb)

    def satisfy_constraint(self, measured: dict, target: dict):
        for key in measured:
            # if the constraint is not specified, we just continue
            if key not in target:
                continue
            # if we exceed the constraint, just return false.
            if measured[key] > target[key]:
                return False
        # no constraint violated, return true.
        return True
    
# TODO: implement accuracy predictor
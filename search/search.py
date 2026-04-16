from abc import ABC, abstractmethod

import torch

from utils.measurements import get_flops, get_peak_memory

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
    def __init__(self, net):
        self.net = net

    def get_efficiency(self, spec: dict):
        self.net.set_active_subnet(**spec)
        subnet = self.net.get_active_subnet()
        if torch.cuda.is_available():
            subnet = subnet.cuda()
        ############### YOUR CODE STARTS HERE ###############
        # Hint: take a look at the `evaluate_sub_network` function above.
        # Hint: the data shape is (batch_size, input_channel, image_size, image_size)
        data_shape = (1, 3, image_size, image_size)
        macs = get_flops(subnet, data_shape)
        peak_memory = get_peak_memory(subnet, data_shape)
        ################ YOUR CODE ENDS HERE ################

        return dict(millionMACs=macs / 1e6, KBPeakMemory=peak_memory / 1024)

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
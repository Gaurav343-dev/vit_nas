from abc import ABC, abstractmethod
import copy
import random

import torch

from utils.measurements import get_macs, get_peak_memory


class SearchSpace:
    def __init__(
        self,
        embed_dim_options: list,
        num_heads_options: list,
        mlp_dim_options: list,
        num_layers_options: list,
    ):
        assert len(embed_dim_options) > 0,  "embed_dim_options must not be empty"
        assert len(num_heads_options) > 0,  "num_heads_options must not be empty"
        assert len(mlp_dim_options) > 0,    "mlp_dim_options must not be empty"
        assert len(num_layers_options) > 0, "num_layers_options must not be empty"
        assert all(n > 0 for n in num_layers_options), \
            "all num_layers_options must be positive integers"

        self.embed_dim_options  = embed_dim_options
        self.num_heads_options  = num_heads_options
        self.mlp_dim_options    = mlp_dim_options
        self.num_layers_options = num_layers_options

    def validate_config(self, config: dict):
        """Raise AssertionError if config is inconsistent with the search space."""
        L = config.get("num_layers")
        E = config.get("embed_dim")
        H = config.get("num_heads")
        M = config.get("mlp_dim")

        assert L is not None, "config must include 'num_layers'"
        assert E is not None, "config must include 'embed_dim'"
        assert H is not None, "config must include 'num_heads'"
        assert M is not None, "config must include 'mlp_dim'"

        assert L in self.num_layers_options, \
            f"num_layers={L} not in num_layers_options={self.num_layers_options}"
        assert E in self.embed_dim_options, \
            f"embed_dim={E} not in embed_dim_options={self.embed_dim_options}"
        assert isinstance(H, list) and len(H) == L, \
            f"num_heads must be a list of length {L}, got {H!r}"
        assert isinstance(M, list) and len(M) == L, \
            f"mlp_dim must be a list of length {L}, got {M!r}"
        assert all(h in self.num_heads_options for h in H), \
            f"some num_heads values not in num_heads_options={self.num_heads_options}: {H}"
        assert all(m in self.mlp_dim_options for m in M), \
            f"some mlp_dim values not in mlp_dim_options={self.mlp_dim_options}: {M}"

    @property
    def size(self) -> int:
        """Total number of distinct subnet configurations in the search space."""
        n_h = len(self.num_heads_options)
        n_m = len(self.mlp_dim_options)
        n_e = len(self.embed_dim_options)
        return n_e * sum(
            (n_h ** L) * (n_m ** L) for L in self.num_layers_options
        )

    def get_max_config(self) -> dict:
        L = max(self.num_layers_options)
        return {
            "embed_dim":  max(self.embed_dim_options),
            "num_heads":  [max(self.num_heads_options)] * L,
            "mlp_dim":    [max(self.mlp_dim_options)]   * L,
            "num_layers": L,
        }

    def get_min_config(self) -> dict:
        L = min(self.num_layers_options)
        return {
            "embed_dim":  min(self.embed_dim_options),
            "num_heads":  [min(self.num_heads_options)] * L,
            "mlp_dim":    [min(self.mlp_dim_options)]   * L,
            "num_layers": L,
        }

    def sample_random_config(self) -> dict:
        L = random.choice(self.num_layers_options)
        return {
            "embed_dim":  random.choice(self.embed_dim_options),
            "num_heads":  [random.choice(self.num_heads_options) for _ in range(L)],
            "mlp_dim":    [random.choice(self.mlp_dim_options)   for _ in range(L)],
            "num_layers": L,
        }

    def mutate_config(self, config: dict, mutate_prob: float = 0.2) -> dict:
        """Return a new config by independently re-sampling each field with mutate_prob.

        If num_layers changes, per-layer lists are resized: existing values kept for
        shared positions, new positions filled with a random valid choice.
        """
        new_cfg = copy.deepcopy(config)

        if random.random() < mutate_prob:
            new_cfg["embed_dim"] = random.choice(self.embed_dim_options)
        if random.random() < mutate_prob:
            new_cfg["num_layers"] = random.choice(self.num_layers_options)

        L = new_cfg["num_layers"]
        heads = (new_cfg["num_heads"] + [random.choice(self.num_heads_options)] * L)[:L]
        mlps  = (new_cfg["mlp_dim"]   + [random.choice(self.mlp_dim_options)]   * L)[:L]
        new_cfg["num_heads"] = [
            random.choice(self.num_heads_options) if random.random() < mutate_prob else h
            for h in heads
        ]
        new_cfg["mlp_dim"] = [
            random.choice(self.mlp_dim_options) if random.random() < mutate_prob else m
            for m in mlps
        ]
        return new_cfg

    def crossover_config(self, config1: dict, config2: dict) -> dict:
        """Return a new config by element-wise crossover of two parent configs.

        Uses the deeper parent as the base (MCUFormer style): positions beyond
        the shorter parent's length are inherited unchanged from the deeper parent.
        """
        if config2["num_layers"] > config1["num_layers"]:
            config1, config2 = config2, config1

        new_cfg    = copy.deepcopy(config1)
        short_L    = config2["num_layers"]

        new_cfg["embed_dim"]  = random.choice([config1["embed_dim"],  config2["embed_dim"]])
        new_cfg["num_layers"] = random.choice([config1["num_layers"], config2["num_layers"]])

        for i in range(short_L):
            new_cfg["num_heads"][i] = random.choice([config1["num_heads"][i], config2["num_heads"][i]])
            new_cfg["mlp_dim"][i]   = random.choice([config1["mlp_dim"][i],   config2["mlp_dim"][i]])

        L = new_cfg["num_layers"]
        new_cfg["num_heads"] = new_cfg["num_heads"][:L]
        new_cfg["mlp_dim"]   = new_cfg["mlp_dim"][:L]
        return new_cfg

    def set_training_dim(self, key, value):
        if key == "embed_dim":
            self.embed_dim_options = [value]
        elif key == "num_heads":
            self.num_heads_options = [value]
        elif key == "mlp_dim":
            self.mlp_dim_options = [value]
        elif key == "num_layers":
            self.num_layers_options = [value]
        else:
            raise ValueError(f"Invalid key: {key}")


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
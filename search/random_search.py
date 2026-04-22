import copy
import torch
from tqdm import tqdm
from train_supernet import SearchSpace
from modules.super_net import SuperNet


class RandomSearcher:
    def __init__(self, efficiency_predictor, search_space: SearchSpace, model: SuperNet, accuracy_predictor=None):
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor  # TODO: use when accuracy predictor is implemented
        self.search_space = search_space
        self.model = model

    def random_valid_sample(self, constraint):
        # randomly sample configs until finding one that satisfies the constraint
        while True:
            # sample = self.accuracy_predictor.arch_encoder.random_sample_arch()
            sample_config = self.search_space.sample_random_config()
            self.model.set_active_subnet(sample_config)
            efficiency = self.efficiency_predictor.get_efficiency(self.model)
            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                # deepcopy config so each entry in the pool is independent
                return copy.deepcopy(sample_config), efficiency

    def run_search(self, constraint, n_subnets=100):
        subnet_pool = []  # list of (config, efficiency) tuples
        # sample subnets
        for _ in tqdm(range(n_subnets)):
            sample_config, efficiency = self.random_valid_sample(constraint)
            subnet_pool.append((sample_config, efficiency))

        if self.accuracy_predictor is not None:
            # TODO: wire up fully when accuracy predictor is implemented
            accs = self.accuracy_predictor.predict_acc([s[0] for s in subnet_pool])
            best_idx = torch.argmax(accs)
            return subnet_pool[best_idx], subnet_pool

        # no accuracy predictor — return the most efficient (lowest MACs) valid config
        best_config, best_efficiency = min(subnet_pool, key=lambda x: x[1]["millionMACs"])
        return (best_config, best_efficiency), subnet_pool
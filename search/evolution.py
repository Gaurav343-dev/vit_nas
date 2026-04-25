import copy
import numpy as np
from tqdm import tqdm

from modules.super_net import SuperNet
from train_supernet import SearchSpace


class EvolutionSearcher:
    def __init__(self, efficiency_predictor, model: SuperNet, search_space: SearchSpace,
                 accuracy_predictor=None, **kwargs):
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor  # optional; –MACs used as proxy if absent
        self.model = model
        self.search_space = search_space

        self.arch_mutate_prob = kwargs.get("arch_mutate_prob", 0.2)
        self.population_size  = kwargs.get("population_size",  50)
        self.max_time_budget  = kwargs.get("max_time_budget",  20)
        self.parent_ratio     = kwargs.get("parent_ratio",     0.25)
        self.mutation_ratio   = kwargs.get("mutation_ratio",   0.5)

    def update_hyper_params(self, new_param_dict):
        self.__dict__.update(new_param_dict)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _get_efficiency(self, config: dict) -> dict:
        self.model.set_active_subnet(config)
        return self.efficiency_predictor.get_efficiency(self.model)

    def _satisfies(self, config: dict, constraint: dict) -> tuple:
        eff = self._get_efficiency(config)
        return self.efficiency_predictor.satisfy_constraint(eff, constraint), eff

    def _score_pool(self, configs: list) -> list:
        """Score each config (higher = better).
        Uses accuracy_predictor when available, otherwise –MACs as a proxy."""
        if self.accuracy_predictor is not None:
            return self.accuracy_predictor.predict_acc(configs)
        return [-self._get_efficiency(cfg)["millionMACs"] for cfg in configs]

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def random_valid_sample(self, constraint: dict):
        while True:
            config = self.search_space.sample_random_config()
            ok, eff = self._satisfies(config, constraint)
            if ok:
                return copy.deepcopy(config), eff

    def mutate_sample(self, config: dict, constraint: dict):
        while True:
            new_cfg = self.search_space.mutate_config(config, self.arch_mutate_prob)
            ok, eff = self._satisfies(new_cfg, constraint)
            if ok:
                return new_cfg, eff

    def crossover_sample(self, config1: dict, config2: dict, constraint: dict):
        while True:
            new_cfg = self.search_space.crossover_config(config1, config2)
            ok, eff = self._satisfies(new_cfg, constraint)
            if ok:
                return new_cfg, eff

    # ------------------------------------------------------------------
    # main search loop
    # ------------------------------------------------------------------
    def run_search(self, constraint: dict, **kwargs):
        self.update_hyper_params(kwargs)

        mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        parents_size     = int(round(self.parent_ratio   * self.population_size))

        best_valids = [-1e9]
        best_info   = None       # (score, config)
        population  = []         # list of (score, config)

        # --- seed population ---
        print(f"Seeding population ({self.population_size} subnets) …")
        child_pool = []
        for _ in tqdm(range(self.population_size)):
            cfg, _ = self.random_valid_sample(constraint)
            child_pool.append(cfg)

        scores = self._score_pool(child_pool)
        population = list(zip(scores, child_pool))

        # --- evolutionary loop ---
        with tqdm(total=self.max_time_budget, desc="Evolving") as t:
            for _ in range(self.max_time_budget):
                population = sorted(population, key=lambda x: x[0], reverse=True)
                population = population[:parents_size]

                best_score = population[0][0]
                if best_score > best_valids[-1]:
                    best_valids.append(best_score)
                    best_info = population[0]
                else:
                    best_valids.append(best_valids[-1])

                child_pool = []
                for _ in range(mutation_numbers):
                    par = population[np.random.randint(parents_size)][1]
                    new_cfg, _ = self.mutate_sample(par, constraint)
                    child_pool.append(new_cfg)

                for _ in range(self.population_size - mutation_numbers):
                    p1 = population[np.random.randint(parents_size)][1]
                    p2 = population[np.random.randint(parents_size)][1]
                    new_cfg, _ = self.crossover_sample(p1, p2, constraint)
                    child_pool.append(new_cfg)

                scores = self._score_pool(child_pool)
                population.extend(zip(scores, child_pool))

                t.update(1)

        return best_info  # (score, best_config)

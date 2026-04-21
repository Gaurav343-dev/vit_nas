import torch
from tqdm import tqdm
from train_supernet import SearchSpace
from modules.super_net import SuperNet


class RandomSearcher:
    def __init__(self, efficiency_predictor, accuracy_predictor, search_space: SearchSpace, model: SuperNet):
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.search_space = search_space
        self.model = model

    def random_valid_sample(self, constraint):
        # randomly sample subnets until finding one that satisfies the constraint
        while True:
            # sample = self.accuracy_predictor.arch_encoder.random_sample_arch()
            sample_config = self.search_space.sample_random_config()
            self.model.set_active_subnet(sample_config)
            efficiency = self.efficiency_predictor.get_efficiency(self.model)
            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                return self.model, efficiency

    def run_search(self, constraint, n_subnets=100):
        subnet_pool = []
        # sample subnets
        for _ in tqdm(range(n_subnets)):
            sample, efficiency = self.random_valid_sample(constraint)
            subnet_pool.append(sample)
        # predict the accuracy of subnets
        accs = self.accuracy_predictor.predict_acc(subnet_pool)
        ############### YOUR CODE STARTS HERE ###############
        # hint: one line of code
        # get the index of the best subnet
        best_idx = torch.argmax(accs)
        ############### YOUR CODE ENDS HERE #################
        # return the best subnet
        return accs[best_idx], subnet_pool[best_idx]
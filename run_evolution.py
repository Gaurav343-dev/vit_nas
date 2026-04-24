"""Run evolutionary architecture search over a trained supernet.

Usage:
    python run_evolution.py --checkpoint final_supernet.pth --mac-constraint 600 --n-subnets 20 --n-generations 10
"""
import argparse
import json
import random

import numpy as np
import torch
from torch import nn

from eval import evaluate
from modules.super_net import SuperNet
from search.evolution import EvolutionSearcher
from search.search import AnalyticalEfficiencyPredictor
from train_supernet import SearchSpace
from utils.data_handler import build_dataloader
from utils.measurements import get_parameters_size


def parse_args():
    parser = argparse.ArgumentParser(description="Evolutionary NAS search over trained supernet")

    # supernet architecture (must match the trained checkpoint)
    parser.add_argument("--img-size",    type=int,   default=32)
    parser.add_argument("--patch-size",  type=int,   default=4)
    parser.add_argument("--embed-dim",   type=int,   default=512)
    parser.add_argument("--num-layers",  type=int,   default=6)
    parser.add_argument("--num-heads",   type=int,   default=8)
    parser.add_argument("--mlp-dim",     type=int,   default=1024)
    parser.add_argument("--num-classes", type=int,   default=10)
    parser.add_argument("--dropout",     type=float, default=0.1)

    # search space
    parser.add_argument("--embed-dim-options",  type=int, nargs="+", default=[512])
    parser.add_argument("--num-heads-options",  type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--mlp-dim-options",    type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--num-layers-options", type=int, nargs="+", default=[2, 4, 6])

    # constraints (omit to disable)
    parser.add_argument("--mac-constraint",         type=float, default=None, help="Max MACs in millions")
    parser.add_argument("--peak-memory-constraint", type=float, default=None, help="Max peak activation memory in KB")

    # evolution hyper-parameters
    parser.add_argument("--population-size",  type=int,   default=50)
    parser.add_argument("--n-generations",    type=int,   default=20,  help="Evolution time budget (generations)")
    parser.add_argument("--parent-ratio",     type=float, default=0.25)
    parser.add_argument("--mutation-ratio",   type=float, default=0.5)
    parser.add_argument("--mutate-prob",      type=float, default=0.2)

    parser.add_argument("--checkpoint",   type=str, default=None)
    parser.add_argument("--batch-size",   type=int, default=128)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--results-json", type=str, default="evolution_results.json")

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SuperNet(
        img_size=args.img_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, mlp_dim=args.mlp_dim,
        num_classes=args.num_classes, dropout=args.dropout,
    ).to(device)

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Warning: no checkpoint — searching over randomly initialised supernet.")

    model.eval()

    search_space = SearchSpace(
        embed_dim_options=args.embed_dim_options,
        num_heads_options=args.num_heads_options,
        mlp_dim_options=args.mlp_dim_options,
        num_layers_options=args.num_layers_options,
    )
    print(f"Search space size: {search_space.size:.1e}")

    constraint = {}
    if args.mac_constraint is not None:
        constraint["millionMACs"] = args.mac_constraint
    if args.peak_memory_constraint is not None:
        constraint["KBPeakMemory"] = args.peak_memory_constraint
    print(f"Constraint: {constraint}")

    efficiency_predictor = AnalyticalEfficiencyPredictor(model, img_size=args.img_size)
    searcher = EvolutionSearcher(
        efficiency_predictor, model, search_space,
        population_size=args.population_size,
        max_time_budget=args.n_generations,
        parent_ratio=args.parent_ratio,
        mutation_ratio=args.mutation_ratio,
        arch_mutate_prob=args.mutate_prob,
    )

    best_score, best_config = searcher.run_search(constraint)

    model.set_active_subnet(best_config)
    eff = efficiency_predictor.get_efficiency(model)
    subnet = model.get_active_subnet().to(device)

    print("\n=== Best subnet found ===")
    print(f"  Config: {best_config}")
    print(f"  MACs:   {eff['millionMACs']:.2f}M")
    print(f"  Params: {get_parameters_size(subnet)}")

    print("\nEvaluating on CIFAR-10 test set...")
    _, test_loader, _ = build_dataloader(batch_size=args.batch_size, img_size=args.img_size)
    test_loss, test_acc = evaluate(subnet, test_loader, nn.CrossEntropyLoss(), device)

    print(f"\n=== Test Results ===")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_acc * 100:.2f}%")

    result = {
        "config": best_config,
        "millionMACs": eff["millionMACs"],
        "accuracy": test_acc * 100,
    }
    with open(args.results_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {args.results_json}")


if __name__ == "__main__":
    main()

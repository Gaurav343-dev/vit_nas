"""Run random architecture search over a trained supernet.

Usage:
    python run_search.py --checkpoint final_supernet.pth --mac-constraint 200 --n-subnets 100

The script loads a trained supernet, runs random search under the given MACs
constraint, extracts the best subnet as a static model, evaluates it on the
CIFAR-10 test set, and prints the results.
"""
import argparse
import json
import random

import numpy as np
import torch
from torch import nn

from eval import evaluate
from modules.super_net import SuperNet
from search.random_search import RandomSearcher
from search.search import AnalyticalEfficiencyPredictor
from train_supernet import SearchSpace
from utils.data_handler import build_dataloader
from utils.measurements import get_parameters_size


def parse_args():
    parser = argparse.ArgumentParser(description="Random NAS search over trained supernet")

    # supernet architecture (must match the trained checkpoint)
    parser.add_argument("--img-size",    type=int, default=32)
    parser.add_argument("--patch-size",  type=int, default=4)
    parser.add_argument("--embed-dim",   type=int, default=512,  help="Max embed dim of supernet")
    parser.add_argument("--num-layers",  type=int, default=6,    help="Max layers of supernet")
    parser.add_argument("--num-heads",   type=int, default=8,    help="Max heads of supernet")
    parser.add_argument("--mlp-dim",     type=int, default=1024, help="Max MLP dim of supernet")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--dropout",     type=float, default=0.1)

    # search space options (subsets of the supernet's max dims)
    parser.add_argument("--embed-dim-options",  type=int, nargs="+", default=[512])
    parser.add_argument("--num-heads-options",  type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--mlp-dim-options",    type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--num-layers-options", type=int, nargs="+", default=[2, 4, 6])

    # search settings
    parser.add_argument("--checkpoint",     type=str,   default=None, help="Path to supernet .pth checkpoint")
    parser.add_argument("--mac-constraint",        type=float, default=None, help="Max allowed MACs in millions (omit to disable)")
    parser.add_argument("--peak-memory-constraint", type=float, default=None, help="Max allowed peak activation memory in KB (omit to disable)")
    parser.add_argument("--n-subnets",      type=int,   default=100,  help="Number of subnets to sample")
    parser.add_argument("--batch-size",     type=int,   default=128)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--results-json", type=str, default="results.json",
                        help="Path to save all sampled subnet results (for Pareto plot)")

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

    # build supernet
    model = SuperNet(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)

    # load trained weights if provided
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device)
        # checkpoints may be saved as {"model": state_dict, ...} or directly as state_dict
        state_dict = state.get("model", state)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Warning: no checkpoint provided — searching over randomly initialised supernet.")

    model.eval()

    # build search space
    search_space = SearchSpace(
        embed_dim_options=args.embed_dim_options,
        num_heads_options=args.num_heads_options,
        mlp_dim_options=args.mlp_dim_options,
        num_layers_options=args.num_layers_options,
    )

    efficiency_predictor = AnalyticalEfficiencyPredictor(model, img_size=args.img_size)
    searcher = RandomSearcher(efficiency_predictor, search_space, model)

    constraint = {}
    if args.mac_constraint is not None:
        constraint["millionMACs"] = args.mac_constraint
    if args.peak_memory_constraint is not None:
        constraint["KBPeakMemory"] = args.peak_memory_constraint

    print(f"Search space size: {search_space.size:.1e}")
    print(f"\nRunning random search — constraint: {constraint}, n_subnets: {args.n_subnets}")

    (best_config, best_efficiency), subnet_pool = searcher.run_search(constraint, n_subnets=args.n_subnets)

    # save all sampled results to JSON for Pareto plot
    records = [
        {"millionMACs": eff["millionMACs"], "config": cfg}
        for cfg, eff in subnet_pool
    ]
    with open(args.results_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} subnet results → {args.results_json}")

    print("\n=== Best subnet found ===")
    print(f"  Config:     {best_config}")
    print(f"  MACs:       {best_efficiency['millionMACs']:.2f}M")

    # extract static subnet with exactly the searched config's parameter count
    model.set_active_subnet(best_config)
    subnet = model.get_active_subnet().to(device)
    print(f"  Params:     {get_parameters_size(subnet)}")

    # evaluate subnet on CIFAR-10 test set
    print("\nEvaluating subnet on CIFAR-10 test set...")
    _, test_loader, _ = build_dataloader(batch_size=args.batch_size, img_size=args.img_size)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(subnet, test_loader, criterion, device)

    print(f"\n=== Test Results ===")
    print(f"  Loss:       {test_loss:.4f}")
    print(f"  Accuracy:   {test_acc * 100:.2f}%")

    # add accuracy to results and re-save (best subnet accuracy now known)
    records.append({
        "millionMACs": best_efficiency["millionMACs"],
        "accuracy": test_acc * 100,
        "config": best_config,
        "is_best": True,
    })
    with open(args.results_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Updated results with best subnet accuracy → {args.results_json}")


if __name__ == "__main__":
    main()

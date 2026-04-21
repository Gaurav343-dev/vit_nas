"""Run random architecture search over a trained supernet.

Usage:
    python run_search.py --checkpoint final_supernet.pth --mac-constraint 200 --n-subnets 100

The script loads a trained supernet, runs random search under the given MACs
constraint, and prints the best found config with its efficiency stats.
"""
import argparse
import random

import numpy as np
import torch

from modules.super_net import SuperNet
from search.random_search import RandomSearcher
from search.search import AnalyticalEfficiencyPredictor
from train_supernet import SearchSpace


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
    parser.add_argument("--embed-dim-options", type=int, nargs="+", default=[256, 384, 512])
    parser.add_argument("--num-heads-options", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--mlp-dim-options",   type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--num-layers-options", type=int, nargs="+", default=[2, 4, 6])

    # search settings
    parser.add_argument("--checkpoint",     type=str,   default=None, help="Path to supernet .pth checkpoint")
    parser.add_argument("--mac-constraint", type=float, default=200,  help="Max allowed MACs in millions")
    parser.add_argument("--n-subnets",      type=int,   default=100,  help="Number of subnets to sample")
    parser.add_argument("--seed",           type=int,   default=42)

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

    constraint = {"millionMACs": args.mac_constraint}
    print(f"\nRunning random search — constraint: {constraint}, n_subnets: {args.n_subnets}")

    best_config, best_efficiency = searcher.run_search(constraint, n_subnets=args.n_subnets)

    print("\n=== Best subnet found ===")
    print(f"  Config:     {best_config}")
    print(f"  MACs:       {best_efficiency['millionMACs']:.2f}M")


if __name__ == "__main__":
    main()

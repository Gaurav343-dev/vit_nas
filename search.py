"""
search.py

Phase 4 of the NAS pipeline: uses the trained predictor to efficiently
search for the best architecture under a multi-objective cost function.

Strategy:
    1. Sample a large number of random configs from the search space
    2. For each config, collect internal signals from the supernet (one batch)
    3. Use the predictor to estimate accuracy, FLOPs, and params
    4. Score each config with the cost function
    5. Return the top-K architectures

Cost function:
    cost = -accuracy + lambda_flops * (flops / max_flops)
                     + lambda_params * (params / max_params)

    Lower cost = better architecture.
    lambda values control the accuracy vs efficiency tradeoff.

Usage:
    python search.py \
        --supernet-path best_supernet.pth \
        --predictor-path predictor.pt \
        --num-candidates 2000 \
        --lambda-flops 0.3 \
        --lambda-params 0.2 \
        --top-k 5
"""

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.super_net import SuperNet
from utils.data_handler import build_dataloader
from train_supernet import SearchSpace, set_seed
from predictor import Predictor
from collect_predictor_data import InternalSignalCollector


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

def cost_function(
    accuracy: float,
    flops: float,
    params: float,
    max_flops: float,
    max_params: float,
    lambda_flops: float = 0.3,
    lambda_params: float = 0.2,
) -> float:
    """
    Multi-objective cost function balancing accuracy and efficiency.

    Lower is better:
        cost = -accuracy
             + lambda_flops  * (flops  / max_flops)
             + lambda_params * (params / max_params)

    Args:
        accuracy:      predicted validation accuracy [0, 1]
        flops:         predicted FLOPs for this config
        params:        predicted parameter count
        max_flops:     maximum FLOPs across all candidates (for normalization)
        max_params:    maximum params across all candidates (for normalization)
        lambda_flops:  penalty weight for FLOPs
        lambda_params: penalty weight for params
    """
    flops_term  = lambda_flops  * (flops  / max_flops)  if max_flops  > 0 else 0.0
    params_term = lambda_params * (params / max_params) if max_params > 0 else 0.0
    return -accuracy + flops_term + params_term


# ---------------------------------------------------------------------------
# Signal collection (single batch, no grad needed for search)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_signals_no_grad(
    model: SuperNet,
    val_loader,
    device: torch.device,
) -> dict:
    """
    Collect internal signals for the current active subnet using a
    single validation batch. Uses no_grad so it's fast during search.

    Activation norms and attention entropy can be collected without
    gradients; gradient norms are skipped here (set to 0) since we
    don't want the overhead of a backward pass for every candidate.
    """
    model.eval()
    inputs, _ = next(iter(val_loader))
    inputs = inputs.to(device)

    activation_norms = []
    attn_entropies = []

    hooks = []
    active_blocks = model.transformer_blocks[: model.active_num_layers]

    for block in active_blocks:
        def make_act_hook(storage):
            def hook(module, input, output):
                storage.append(output.detach().norm(dim=-1).mean().item())
            return hook

        def make_attn_hook(storage):
            def hook(module, input, output):
                x = input[0].detach()
                B, T, _ = x.shape
                E = module.active_embed_dim
                H = module.active_num_heads
                head_dim = E // H
                module.qkv_linear.active_out = 3 * E
                qkv = module.qkv_linear(x)
                qkv = qkv.reshape(B, T, 3, H, head_dim).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
                attn = F.softmax(q @ k.transpose(-2, -1) * (head_dim ** -0.5), dim=-1)
                entropy = -(attn * (attn + 1e-9).log()).sum(dim=-1).mean().item()
                storage.append(entropy)
            return hook

        hooks.append(block.register_forward_hook(make_act_hook(activation_norms)))
        hooks.append(block.mha.register_forward_hook(make_attn_hook(attn_entropies)))

    model(inputs)

    for h in hooks:
        h.remove()

    def stats(vals):
        if not vals:
            return {"mean": 0.0, "std": 0.0, "max": 0.0}
        arr = np.array(vals)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "max": float(arr.max())}

    return {
        "activation_norm":    stats(activation_norms),
        "gradient_norm":      {"mean": 0.0, "std": 0.0, "max": 0.0},  # skipped in search
        "attention_entropy":  stats(attn_entropies),
    }


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def search(
    model: SuperNet,
    predictor: Predictor,
    search_space: SearchSpace,
    val_loader,
    device: torch.device,
    num_candidates: int = 2000,
    lambda_flops: float = 0.3,
    lambda_params: float = 0.2,
    top_k: int = 5,
) -> list[dict]:
    """
    Sample num_candidates configs, score each with the predictor and
    cost function, return the top_k results sorted by cost (ascending).

    Each result dict contains:
        config, predicted_accuracy, predicted_flops, predicted_params, cost
    """
    # always include max and min configs
    configs = [search_space.get_max_config(), search_space.get_min_config()]
    configs += [search_space.sample_random_config() for _ in range(num_candidates - 2)]

    # deduplicate
    seen = set()
    unique_configs = []
    for cfg in configs:
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)

    print(f"Evaluating {len(unique_configs)} unique configs with predictor...")

    results = []
    for config in tqdm(unique_configs, desc="Searching"):
        model.set_active_subnet(config)

        # collect signals for this config
        signals = collect_signals_no_grad(model, val_loader, device)

        # build record for predictor
        record = {"config": config, "signals": signals}

        # predict with MLP
        acc, flops, params = predictor.predict(record)

        results.append({
            "config":              config,
            "predicted_accuracy":  acc,
            "predicted_flops":     flops,
            "predicted_params":    params,
            # cost computed after we know max_flops / max_params
        })

    # normalize flops and params across all candidates
    max_flops  = max(r["predicted_flops"]  for r in results)
    max_params = max(r["predicted_params"] for r in results)

    # score each candidate
    for r in results:
        r["cost"] = cost_function(
            accuracy=r["predicted_accuracy"],
            flops=r["predicted_flops"],
            params=r["predicted_params"],
            max_flops=max_flops,
            max_params=max_params,
            lambda_flops=lambda_flops,
            lambda_params=lambda_params,
        )

    # sort by cost ascending (lower = better)
    results.sort(key=lambda r: r["cost"])
    return results[:top_k]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NAS search using trained predictor")
    parser.add_argument("--supernet-path",  type=str, required=True,
                        help="Path to trained supernet checkpoint")
    parser.add_argument("--predictor-path", type=str, required=True,
                        help="Path to trained predictor (.pt)")
    parser.add_argument("--num-candidates", type=int, default=2000,
                        help="Number of random configs to score")
    parser.add_argument("--lambda-flops",   type=float, default=0.3,
                        help="Penalty weight for FLOPs in cost function")
    parser.add_argument("--lambda-params",  type=float, default=0.2,
                        help="Penalty weight for params in cost function")
    parser.add_argument("--top-k",          type=int,   default=5,
                        help="Number of top architectures to return")
    parser.add_argument("--output",         type=str,   default="search_results.pt",
                        help="Where to save search results")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--batch-size",     type=int,   default=128)
    parser.add_argument("--img-size",       type=int,   default=32)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # search space — must match supernet training
    # to this
    search_space = SearchSpace(
        embed_dim_options=[512],
        num_heads_options=[2, 4, 8],
        mlp_dim_options=[1024],
        num_layers_options=[6],
    )

    # load supernet
    max_config = search_space.get_max_config()
    model = SuperNet(
        img_size=args.img_size,
        patch_size=4,
        embed_dim=max_config["embed_dim"],
        num_layers=max_config["num_layers"],
        num_heads=max_config["num_heads"],
        mlp_dim=max_config["mlp_dim"],
        num_classes=10,
        dropout=0.0,
    )
    model.load_state_dict(torch.load(args.supernet_path, map_location=device))
    model.to(device)

    # load predictor
    predictor = Predictor.load(args.predictor_path)
    print(f"Loaded predictor from {args.predictor_path}")

    # load val set for signal collection
    _, _, val_loader = build_dataloader(
        batch_size=args.batch_size,
        validation_split=0.1,
        img_size=args.img_size,
    )

    # run search
    top_configs = search(
        model=model,
        predictor=predictor,
        search_space=search_space,
        val_loader=val_loader,
        device=device,
        num_candidates=args.num_candidates,
        lambda_flops=args.lambda_flops,
        lambda_params=args.lambda_params,
        top_k=args.top_k,
    )

    # print results
    print(f"\nTop {args.top_k} architectures:")
    print("-" * 70)
    for i, r in enumerate(top_configs):
        print(f"\nRank {i+1}:")
        print(f"  Config:    {r['config']}")
        print(f"  Accuracy:  {r['predicted_accuracy']:.4f}")
        print(f"  FLOPs:     {r['predicted_flops']:,.0f}")
        print(f"  Params:    {r['predicted_params']:,.0f}")
        print(f"  Cost:      {r['cost']:.6f}")

    # save results
    torch.save(top_configs, args.output)
    print(f"\nSaved results to {args.output}")

    # print the single best config cleanly
    best = top_configs[0]
    print(f"\nBest config: {best['config']}")
    print(f"  Predicted accuracy: {best['predicted_accuracy']:.4f}")
    print(f"  FLOPs:              {best['predicted_flops']:,.0f}")
    print(f"  Params:             {best['predicted_params']:,.0f}")


if __name__ == "__main__":
    main()
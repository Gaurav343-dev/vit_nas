"""Visualization utilities for vit_nas NAS results.

Two plots:
  1. architecture_heatmap(config)  — per-layer heads + MLP dim as a colored grid
  2. pareto_plot(results)          — accuracy vs MACs scatter with Pareto frontier

Usage (standalone):
    python visualize.py --mode heatmap \
        --num-layers 6 --num-heads 2 8 4 8 2 4 --mlp-dim 256 1024 512 1024 256 512

    python visualize.py --mode pareto --results-json results.json \
        --checkpoint supernet.pth
"""
import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# 1. Architecture heatmap
# ---------------------------------------------------------------------------

def architecture_heatmap(config: dict, save_path: str = None, ax=None):
    """Draw a per-layer architecture heatmap for a single subnet config.

    Rows: heads, MLP dim (two properties per layer)
    Columns: layer index

    Args:
        config: dict with keys embed_dim, num_layers, num_heads (list), mlp_dim (list)
        save_path: if given, save the figure to this path
        ax: optional existing axes to draw into
    """
    L = config["num_layers"]
    heads = config["num_heads"]    # list[int]
    mlps  = config["mlp_dim"]      # list[int]
    embed = config["embed_dim"]

    # normalise to [0,1] for colour mapping
    all_heads = sorted(set(heads))
    all_mlps  = sorted(set(mlps))

    def norm(val, options):
        idx = options.index(val)
        return idx / max(len(options) - 1, 1)

    grid = np.zeros((2, L))
    for i in range(L):
        grid[0, i] = norm(heads[i], all_heads)
        grid[1, i] = norm(mlps[i],  all_mlps)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(6, L * 0.9), 3))

    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    # row labels
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["num_heads", "mlp_dim"], fontsize=11)

    # column labels
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"L{i}" for i in range(L)], fontsize=10)
    ax.set_xlabel("Layer", fontsize=11)

    # annotate cells with actual values
    for i in range(L):
        ax.text(i, 0, str(heads[i]), ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(i, 1, str(mlps[i]),  ha="center", va="center", fontsize=10, fontweight="bold")

    title = f"Subnet Architecture  |  embed_dim={embed}, L={L}"
    ax.set_title(title, fontsize=12, pad=8)

    plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.03, pad=0.02,
                 label="relative option rank")

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved heatmap → {save_path}")
        else:
            plt.show()
        plt.close()

def _evaluate_accuracy(config: dict, checkpoint: str,
                        img_size: int = 32, batch_size: int = 256) -> float:
    """Extract subnet for config from checkpoint and evaluate on CIFAR-10 test set."""
    import torch
    from torch import nn
    from modules.super_net import SuperNet
    from utils.data_handler import build_dataloader
    from eval import evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # infer supernet max dims from the config (embed is always the max)
    E = config["embed_dim"]
    state = torch.load(checkpoint, map_location=device)
    # derive max heads/mlp from checkpoint key shapes
    # qkv weight shape: (3*max_E, max_E) in first block
    sd = state.get("model", state)
    max_qkv_out = sd["transformer_blocks.0.mha.qkv_linear.linear.weight"].shape[0]
    max_E = max_qkv_out // 3
    max_mlp = sd["transformer_blocks.0.mlp.fc1.linear.weight"].shape[0]
    max_L = sum(1 for k in sd if k.startswith("transformer_blocks.") and k.endswith(".mha.qkv_linear.linear.weight"))
    max_H = sd["transformer_blocks.0.mha.qkv_linear.linear.weight"].shape[0] // 3 // (max_E // max_E)
    # simpler: infer num_heads from max_E and head_dim stored implicitly
    # just use max values from config lists
    max_H = max(config["num_heads"])

    model = SuperNet(img_size=img_size, patch_size=4,
                     embed_dim=max_E, num_layers=max_L,
                     num_heads=max_H, mlp_dim=max_mlp,
                     num_classes=10, dropout=0.1)
    model.load_state_dict(sd)
    model.to(device).eval()

    model.set_active_subnet(config)
    subnet = model.get_active_subnet().to(device)

    _, test_loader, _ = build_dataloader(batch_size=batch_size, img_size=img_size)
    _, acc = evaluate(subnet, test_loader, nn.CrossEntropyLoss(), device)
    return acc * 100


def _fill_missing_accuracies(results: list, checkpoint: str,
                              img_size: int = 32, batch_size: int = 256) -> list:
    """Evaluate accuracy for any result entry that is missing it."""
    missing = [r for r in results if "accuracy" not in r]
    if not missing:
        return results
    print(f"Evaluating accuracy for {len(missing)} subnets (this may take a while)...")
    from tqdm import tqdm
    for r in tqdm(missing):
        r["accuracy"] = _evaluate_accuracy(
            r["config"], checkpoint, img_size=img_size, batch_size=batch_size
        )
    return results


def _pareto_frontier(macs, accs):
    """Return indices of Pareto-optimal points (minimise MACs, maximise acc)."""
    points = sorted(zip(macs, accs, range(len(macs))), key=lambda x: x[0])
    pareto_idx = []
    best_acc = -1
    for m, a, i in points:
        if a > best_acc:
            best_acc = a
            pareto_idx.append(i)
    return pareto_idx


def pareto_plot(results: list, save_path: str = None, ax=None,
                highlight_config: dict = None, checkpoint: str = None,
                img_size: int = 32, batch_size: int = 256):
    """Scatter plot of accuracy vs MACs with the Pareto frontier highlighted.

    Args:
        results: list of dicts with keys: "millionMACs", "config",
                 and optionally "accuracy". Missing accuracy is evaluated
                 on-the-fly if checkpoint is provided.
        checkpoint: path to supernet .pth — required when any result lacks accuracy
        save_path: if given, save the figure here
        ax: optional existing axes
        highlight_config: optionally highlight a specific result by config match
    """
    # fill missing accuracies if checkpoint provided
    if checkpoint and any("accuracy" not in r for r in results):
        results = _fill_missing_accuracies(results, checkpoint,
                                           img_size=img_size, batch_size=batch_size)
        # persist updated results back to JSON if a path is known
        if save_path and save_path.endswith(".json"):
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)

    # filter to only entries that have accuracy (skip any still missing)
    results = [r for r in results if "accuracy" in r]
    if not results:
        print("No accuracy data available — cannot draw Pareto plot.")
        return
    macs = [r["millionMACs"] for r in results]
    accs = [r["accuracy"] for r in results]

    # normalise accuracy to percent if needed
    if max(accs) <= 1.0:
        accs = [a * 100 for a in accs]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    # all sampled subnets
    ax.scatter(macs, accs, alpha=0.5, s=30, color="steelblue", label="sampled subnets", zorder=2)

    # Pareto frontier
    pidx = _pareto_frontier(macs, accs)
    pmacs = [macs[i] for i in pidx]
    paccs = [accs[i] for i in pidx]
    # sort by MACs for line
    paired = sorted(zip(pmacs, paccs))
    pmacs_s, paccs_s = zip(*paired)
    ax.plot(pmacs_s, paccs_s, "o-", color="tomato", linewidth=2, markersize=6,
            label="Pareto frontier", zorder=3)

    # optional highlight
    if highlight_config is not None:
        for r, m, a in zip(results, macs, accs):
            if r.get("config") == highlight_config:
                ax.scatter([m], [a], s=120, color="gold", edgecolors="black",
                           zorder=4, label="selected subnet")
                break

    ax.set_xlabel("MACs (M)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("NAS Results: Accuracy vs MACs", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved pareto plot → {save_path}")
        else:
            plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="vit_nas visualizations")
    parser.add_argument("--mode", choices=["heatmap", "pareto", "both"], default="heatmap")

    # heatmap args
    parser.add_argument("--embed-dim",  type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads",  type=int, nargs="+", default=None)
    parser.add_argument("--mlp-dim",    type=int, nargs="+", default=None)

    # pareto args
    parser.add_argument("--results-json", type=str, default=None,
                        help="JSON file: list of {millionMACs, accuracy, config}")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Supernet checkpoint for evaluating missing accuracies")
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--save", type=str, default=None, help="Output file path")
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.mode in ("heatmap", "both"):
        L = args.num_layers
        H = args.num_heads or [8] * L
        M = args.mlp_dim   or [1024] * L
        assert len(H) == L and len(M) == L, \
            "--num-heads and --mlp-dim must each have num-layers values"
        config = {"embed_dim": args.embed_dim, "num_layers": L,
                  "num_heads": H, "mlp_dim": M}

        if args.mode == "both" and args.results_json:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            architecture_heatmap(config, ax=axes[0])
            results = json.load(open(args.results_json))
            pareto_plot(results, ax=axes[1], checkpoint=args.checkpoint,
                        batch_size=args.batch_size)
            plt.tight_layout()
            if args.save:
                plt.savefig(args.save, dpi=150, bbox_inches="tight")
                print(f"Saved → {args.save}")
            else:
                plt.show()
            plt.close()
        else:
            architecture_heatmap(config, save_path=args.save)

    elif args.mode == "pareto":
        assert args.results_json, "--results-json required for pareto mode"
        results = json.load(open(args.results_json))
        pareto_plot(results, save_path=args.save, checkpoint=args.checkpoint,
                    batch_size=args.batch_size)


if __name__ == "__main__":
    main()

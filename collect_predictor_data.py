"""
collect_predictor_data.py

Phase 2 of the NAS pipeline: after the supernet is trained, this script
samples N architecture configs, evaluates each subnet on the validation set
using shared supernet weights, and records:
  - architecture config  (embed_dim, num_heads, mlp_dim, num_layers)
  - internal signals     (per-block activation norms, gradient norms, attention entropy)
  - validation accuracy
  - FLOPs
  - parameter count

Output: a .pt file (list of dicts) that serves as the training dataset
for the predictor model.

Usage:
    python collect_predictor_data.py \
        --supernet-path best_supernet.pth \
        --num-samples 500 \
        --output predictor_dataset.pt
"""

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.super_net import SuperNet
from utils.data_handler import build_dataloader
from train_supernet import set_seed

class SearchSpace:
    def __init__(self, embed_dim_options, num_heads_options, mlp_dim_options, num_layers_options):
        self.embed_dim_options = embed_dim_options
        self.num_heads_options = num_heads_options
        self.mlp_dim_options = mlp_dim_options
        self.num_layers_options = num_layers_options

    def get_max_config(self):
        L = max(self.num_layers_options)
        return {
            "embed_dim": max(self.embed_dim_options),
            "num_heads": [max(self.num_heads_options)] * L,
            "mlp_dim": [max(self.mlp_dim_options)] * L,
            "num_layers": L,
        }

    def get_min_config(self):
        L = min(self.num_layers_options)
        return {
            "embed_dim": min(self.embed_dim_options),
            "num_heads": [min(self.num_heads_options)] * L,
            "mlp_dim": [min(self.mlp_dim_options)] * L,
            "num_layers": L,
        }

    def sample_random_config(self):
        L = random.choice(self.num_layers_options)
        return {
            "embed_dim": random.choice(self.embed_dim_options),
            "num_heads": [random.choice(self.num_heads_options) for _ in range(L)],
            "mlp_dim": [random.choice(self.mlp_dim_options) for _ in range(L)],
            "num_layers": L,
        }
# ---------------------------------------------------------------------------
# FLOPs / parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: SuperNet) -> int:
    total = 0
    E = model.active_embed_dim[0] if isinstance(model.active_embed_dim, list) else model.active_embed_dim
    L = model.active_num_layers[0] if isinstance(model.active_num_layers, list) else model.active_num_layers
    num_heads = model.active_num_heads if isinstance(model.active_num_heads, list) else [model.active_num_heads] * L
    mlp_dims = model.active_mlp_dim if isinstance(model.active_mlp_dim, list) else [model.active_mlp_dim] * L

    # patch embed: Conv2d(3, E, patch_size, patch_size)
    patch_size = model.patch_embed.patch_size
    total += 3 * E * patch_size * patch_size + E

    # cls token + pos encoding
    num_patches = (model.patch_embed.img_size // patch_size) ** 2
    total += E
    total += (num_patches + 1) * E

    # transformer blocks — per layer using actual dims
    for i in range(L):
        M = mlp_dims[i]
        # LayerNorm 1 & 2
        total += 4 * E
        # QKV linear
        total += 3 * E * E + 3 * E
        # proj linear
        total += E * E + E
        # MLP fc1 and fc2
        total += M * E + M
        total += E * M + E

    # final norm
    total += 2 * E
    # head
    total += E * model.head.out_features + model.head.out_features

    return total

def count_flops(model: SuperNet, img_size: int = 32) -> int:
    E = model.active_embed_dim[0] if isinstance(model.active_embed_dim, list) else model.active_embed_dim
    L = model.active_num_layers[0] if isinstance(model.active_num_layers, list) else model.active_num_layers
    num_heads = model.active_num_heads if isinstance(model.active_num_heads, list) else [model.active_num_heads] * L
    mlp_dims = model.active_mlp_dim if isinstance(model.active_mlp_dim, list) else [model.active_mlp_dim] * L
    patch_size = model.patch_embed.patch_size
    T = (img_size // patch_size) ** 2 + 1

    flops = 2 * 3 * E * patch_size * patch_size * (img_size // patch_size) ** 2

    for i in range(L):
        H = num_heads[i]
        M = mlp_dims[i]
        head_dim = E // H
        flops += 2 * T * E * 3 * E
        flops += 2 * H * T * T * head_dim
        flops += 2 * H * T * T * head_dim
        flops += 2 * T * E * E
        flops += 2 * T * E * M + 2 * T * M * E
        flops += 2 * 5 * T * E

    flops += 5 * T * E
    flops += 2 * E * model.head.out_features
    return flops


# ---------------------------------------------------------------------------
# Internal signal collection via hooks
# ---------------------------------------------------------------------------

class InternalSignalCollector:
    """
    Attaches forward and backward hooks to each DynamicTransformerBlock
    to record per-block activation norms, gradient norms, and attention entropy.

    Usage:
        collector = InternalSignalCollector(model)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        signals = collector.get_signals()
        collector.remove_hooks()
    """

    def __init__(self, model: SuperNet):
        self.model = model
        self._fwd_hooks = []
        self._bwd_hooks = []

        # storage: list of dicts, one per active block
        self._activation_norms: list[float] = []
        self._gradient_norms: list[float] = []
        self._attn_entropies: list[float] = []

        self._register_hooks()

    def _register_hooks(self):
        active_blocks = self.model.transformer_blocks[: self.model.active_num_layers]

        for block in active_blocks:
            # forward hook on block output: records activation norm
            def make_fwd_hook(storage):
                def hook(module, input, output):
                    # output: (B, T, E) — mean L2 norm across batch and tokens
                    storage.append(output.detach().norm(dim=-1).mean().item())
                return hook

            # backward hook on block output gradient: records gradient norm
            def make_bwd_hook(storage):
                def hook(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        storage.append(grad_output[0].detach().norm(dim=-1).mean().item())
                return hook

            fh = block.register_forward_hook(make_fwd_hook(self._activation_norms))
            bh = block.register_full_backward_hook(make_bwd_hook(self._gradient_norms))
            self._fwd_hooks.append(fh)
            self._bwd_hooks.append(bh)

            # attention entropy: hook on the MHA module to intercept softmax(QK^T/sqrt(d))
            def make_attn_hook(storage):
                def hook(module, input, output):
                    # Re-compute attention weights from x (input[0]) to get entropy.
                    # This avoids storing attn weights in the main forward pass.
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
                    # entropy: -sum(p * log(p + eps)) per head per token, then mean
                    entropy = -(attn * (attn + 1e-9).log()).sum(dim=-1).mean().item()
                    storage.append(entropy)
                return hook

            ah = block.mha.register_forward_hook(make_attn_hook(self._attn_entropies))
            self._fwd_hooks.append(ah)

    def get_signals(self) -> dict:
        """Return aggregated signal dict. Call after forward + backward."""
        def safe_stats(vals):
            if not vals:
                return {"mean": 0.0, "std": 0.0, "max": 0.0}
            arr = np.array(vals)
            return {"mean": float(arr.mean()), "std": float(arr.std()), "max": float(arr.max())}

        return {
            "activation_norm": safe_stats(self._activation_norms),
            "gradient_norm": safe_stats(self._gradient_norms),
            "attention_entropy": safe_stats(self._attn_entropies),
        }

    def remove_hooks(self):
        for h in self._fwd_hooks + self._bwd_hooks:
            h.remove()
        self._fwd_hooks.clear()
        self._bwd_hooks.clear()


# ---------------------------------------------------------------------------
# Subnet evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_subnet(model: SuperNet, val_loader, device, max_batches: int = 20) -> float:
    """
    Evaluate a locked subnet on a subset of the validation set.
    max_batches limits evaluation cost during data collection.
    """
    model.eval()
    correct = 0
    total = 0
    for i, (inputs, targets) in enumerate(val_loader):
        if i >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_subnet_with_signals(
    model: SuperNet,
    val_loader,
    device,
    max_batches: int = 5,
) -> tuple[float, dict]:
    model.eval()
    inputs, targets = next(iter(val_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    activation_norms = []
    attn_entropies = []
    hooks = []
    active_blocks = model.transformer_blocks[:model.active_num_layers]

    for block in active_blocks:
        def make_act_hook(storage):
            def hook(module, input, output):
                storage.append(output.detach().norm(dim=-1).mean().item())
            return hook
        hooks.append(block.register_forward_hook(make_act_hook(activation_norms)))

    with torch.no_grad():
        model(inputs)

    for h in hooks:
        h.remove()

    def stats(vals):
        if not vals:
            return {"mean": 0.0, "std": 0.0, "max": 0.0}
        arr = np.array(vals)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "max": float(arr.max())}

    signals = {
        "activation_norm": stats(activation_norms),
        "gradient_norm": {"mean": 0.0, "std": 0.0, "max": 0.0},
        "attention_entropy": {"mean": 0.0, "std": 0.0, "max": 0.0},
    }

    accuracy = evaluate_subnet(model, val_loader, device, max_batches=max_batches)
    return accuracy, signals


# ---------------------------------------------------------------------------
# Main data collection loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_subnet_preloaded(model, preloaded_batches, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in preloaded_batches:
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_subnet_with_signals_preloaded(model, preloaded_batches, signal_batch, device):
    model.train()  # need train mode for backward pass
    criterion = torch.nn.CrossEntropyLoss()
    
    inputs, targets = signal_batch
    
    activation_norms = []
    attn_entropies = []
    gradient_norms = []
    hooks = []
    active_blocks = model.transformer_blocks[:model.active_num_layers]

    for block in active_blocks:
        def make_act_hook(storage):
            def hook(module, input, output):
                storage.append(output.detach().norm(dim=-1).mean().item())
            return hook
        hooks.append(block.register_forward_hook(make_act_hook(activation_norms)))

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
        hooks.append(block.mha.register_forward_hook(make_attn_hook(attn_entropies)))

        def make_bwd_hook(storage):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    storage.append(grad_output[0].detach().norm(dim=-1).mean().item())
            return hook
        hooks.append(block.register_full_backward_hook(make_bwd_hook(gradient_norms)))

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    for h in hooks:
        h.remove()

    def stats(vals):
        if not vals:
            return {"mean": 0.0, "std": 0.0, "max": 0.0}
        arr = np.array(vals)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "max": float(arr.max())}

    signals = {
        "activation_norm":   stats(activation_norms),
        "gradient_norm":     stats(gradient_norms),
        "attention_entropy": stats(attn_entropies),
    }

    accuracy = evaluate_subnet_preloaded(model, preloaded_batches, device)
    return accuracy, signals


def collect_predictor_dataset(
    model: SuperNet,
    search_space: SearchSpace,
    val_loader,
    device,
    num_samples: int = 500,
    max_eval_batches: int = 20,
    img_size: int = 32,
) -> list[dict]:
    dataset = []
    seen_configs = set()

    # preload validation batches into GPU memory once
    print("Preloading validation data into memory...")
    preloaded_batches = []
    for i, (inputs, targets) in enumerate(val_loader):
        if i >= max_eval_batches:
            break
        preloaded_batches.append((inputs.to(device), targets.to(device)))
    signal_batch = preloaded_batches[0]
    print(f"Preloaded {len(preloaded_batches)} batches ({len(preloaded_batches) * inputs.size(0)} images)")

    anchor_configs = [search_space.get_max_config(), search_space.get_min_config()]
    random_configs = [search_space.sample_random_config() for _ in range(num_samples)]
    all_configs = anchor_configs + random_configs

    for config in tqdm(all_configs, desc="Collecting predictor data"):
        config_key = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in config.items()))
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)

        model.set_active_subnet(config)

        accuracy, signals = evaluate_subnet_with_signals_preloaded(
            model, preloaded_batches, signal_batch, device
        )
        flops = count_flops(model, img_size=img_size)
        params = count_parameters(model)

        record = {
            "config": config,
            "accuracy": accuracy,
            "flops": flops,
            "params": params,
            "signals": signals,
        }
        dataset.append(record)

    return dataset


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect predictor training data")
    parser.add_argument("--supernet-path", type=str, required=True,
                        help="Path to trained supernet checkpoint (.pth)")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of random configs to sample")
    parser.add_argument("--output", type=str, default="predictor_dataset.pt",
                        help="Output path for the dataset")
    parser.add_argument("--max-eval-batches", type=int, default=20,
                        help="Max validation batches per subnet (trade speed vs accuracy)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=32)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # must match the search space used during supernet training
    search_space = SearchSpace(
        embed_dim_options=[512],
        num_heads_options=[2, 4, 8],
        mlp_dim_options=[256, 512, 1024],
        num_layers_options=[2, 4, 6],
    )

    max_config = search_space.get_max_config()
    model = SuperNet(
    img_size=args.img_size,
    patch_size=4,
    embed_dim=512,
    num_layers=6,
    num_heads=8,
    mlp_dim=1024,
    num_classes=10,
    dropout=0.0,
)
    model.load_state_dict(torch.load(args.supernet_path, map_location=device))
    model.to(device)

    _, _, val_loader = build_dataloader(
        batch_size=args.batch_size,
        validation_split=0.1,
        img_size=args.img_size,
    )

    dataset = collect_predictor_dataset(
        model=model,
        search_space=search_space,
        val_loader=val_loader,
        device=device,
        num_samples=args.num_samples,
        max_eval_batches=args.max_eval_batches,
        img_size=args.img_size,
    )

    torch.save(dataset, args.output)
    print(f"\nSaved {len(dataset)} records to {args.output}")

    # quick summary
    accs = [r["accuracy"] for r in dataset]
    flops = [r["flops"] for r in dataset]
    print(f"Accuracy  — min: {min(accs):.4f}  max: {max(accs):.4f}  mean: {np.mean(accs):.4f}")
    print(f"FLOPs     — min: {min(flops):,}  max: {max(flops):,}")


if __name__ == "__main__":
    main()
# vit_nas

Neural Architecture Search (NAS) for Vision Transformers (ViT) on CIFAR-10 using a weight-sharing supernet and sandwich training.

## Overview

The supernet shares weights across all subnets. Each subnet is defined by a **per-layer** config:

| Dimension | Options | Scope |
|---|---|---|
| `embed_dim` | `[512]` | global |
| `num_heads` | `[2, 4, 8]` | per layer |
| `mlp_dim` | `[256, 512, 1024]` | per layer |
| `num_layers` | `[2, 4, 6]` | global |

**Total search space:** 538,083 subnets

---

## Project Structure

```
vit_nas/
├── train_supernet.py   # Train the weight-sharing supernet (sandwich rule)
├── eval.py             # Evaluate any subnet config on CIFAR-10
├── run_search.py       # Random search → best subnet → evaluate → save JSON
├── visualize.py        # Architecture heatmap + Pareto frontier plot
├── modules/
│   ├── super_net.py    # SuperNet with dynamic layers + get_active_subnet()
│   ├── dynamic_modules.py  # DynamicLinear, DynamicMHA, DynamicMlp, etc.
│   └── sub_net.py      # Static SubNet (extracted from supernet)
├── search/
│   ├── search.py       # AnalyticalEfficiencyPredictor (MACs-based)
│   └── random_search.py    # RandomSearcher
└── utils/
    ├── measurements.py # get_parameters_size(), get_macs()
    └── data_handler.py # CIFAR-10 dataloaders
```

---

## Step 1 — Train the Supernet

```bash
python train_supernet.py
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--num-epochs` | `200` | Number of training epochs |
| `--num-workers` | auto | DataLoader workers (default: cpu_count − 4) |
| `--resume-path` | `None` | Resume from a checkpoint |
| `--mixup` | `0.8` | Enable Mixup augmentation |
| `--cutmix` | `1.0` | Enable CutMix augmentation |
| `--use-wandb` | off | Enable Weights & Biases logging |

**Example:**
```bash
python train_supernet.py --num-epochs 200 --num-workers 8
```

Saves checkpoint as:
```
best_supernet_embed512_heads8_mlp1024_layers6_epochN_acc0.XX_YYYYMMDD-HHh.pth
```

---

## Step 2 — Evaluate a Subnet

Evaluate any subnet config directly on the CIFAR-10 test set.

```bash
python eval.py --model-path <checkpoint.pth> \
  --num-layers <L> \
  --num-heads <h0 h1 ... hL-1> \
  --mlp-dim <m0 m1 ... mL-1>
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model-path` | `final_supernet.pth` | Supernet checkpoint |
| `--num-layers` | max (6) | Number of active layers |
| `--num-heads` | `[max]*L` | Heads per layer (space-separated list) |
| `--mlp-dim` | `[max]*L` | MLP dim per layer (space-separated list) |
| `--batch-size` | `128` | Evaluation batch size |
| `--eval-teacher` | off | Also evaluate the teacher model |

**Examples:**
```bash
# max subnet
python eval.py --model-path checkpoint.pth \
  --num-layers 6 --num-heads 8 8 8 8 8 8 --mlp-dim 1024 1024 1024 1024 1024 1024

# min-width subnet (6 layers, all small)
python eval.py --model-path checkpoint.pth \
  --num-layers 6 --num-heads 2 2 2 2 2 2 --mlp-dim 256 256 256 256 256 256

# mixed per-layer config
python eval.py --model-path checkpoint.pth \
  --num-layers 6 --num-heads 2 8 4 8 2 4 --mlp-dim 256 1024 512 1024 256 512
```

---

## Step 3 — Run Random Search

Sample subnets randomly under a MACs constraint, pick the best, evaluate it, and save all results to JSON.

```bash
python run_search.py --checkpoint <checkpoint.pth> \
  --mac-constraint <MACs_in_millions> \
  --n-subnets <N>
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `None` | Supernet checkpoint (required) |
| `--mac-constraint` | `200` | Max MACs budget in millions |
| `--n-subnets` | `100` | Number of subnets to sample |
| `--embed-dim-options` | `512` | Embed dim choices (must match training) |
| `--num-layers-options` | `2 4 6` | Layer count choices |
| `--num-heads-options` | `2 4 8` | Heads per layer choices |
| `--mlp-dim-options` | `512 1024` | MLP dim per layer choices |
| `--results-json` | `results.json` | Output file for all sampled results |
| `--batch-size` | `128` | Batch size for final evaluation |
| `--seed` | `42` | Random seed |

**Example (layers fixed to 6 to match current checkpoint):**
```bash
python run_search.py \
  --checkpoint final_supernet_embed512_heads8_mlp1024_layers6_epoch200_acc0.90_20260421-02h.pth \
  --mac-constraint 600 \
  --n-subnets 200 \
  --embed-dim-options 512 \
  --num-layers-options 6 \
  --num-heads-options 2 4 8 \
  --mlp-dim-options 256 512 1024 \
  --results-json results.json
```

Outputs:
- Best subnet config + accuracy printed to console
- `results.json` — all sampled subnets with MACs (+ accuracy for best subnet)

---

## Step 4 — Visualize

### Architecture Heatmap

Shows per-layer `num_heads` and `mlp_dim` as a colour-coded grid.

```bash
python visualize.py --mode heatmap \
  --num-layers <L> \
  --num-heads <h0 h1 ... hL-1> \
  --mlp-dim <m0 m1 ... mL-1> \
  --save arch.png
```

**Example:**
```bash
python visualize.py --mode heatmap \
  --num-layers 6 \
  --num-heads 2 8 4 8 2 4 \
  --mlp-dim 256 1024 512 1024 256 512 \
  --save arch.png
```

### Pareto Plot

Scatter of accuracy vs MACs across all sampled subnets, with the Pareto frontier highlighted. Evaluates accuracy on-the-fly for any subnet missing it.

```bash
python visualize.py --mode pareto \
  --results-json results.json \
  --checkpoint <checkpoint.pth> \
  --save pareto.png
```

### Both Side-by-Side

```bash
python visualize.py --mode both \
  --num-layers 6 \
  --num-heads 2 8 4 8 2 4 \
  --mlp-dim 256 1024 512 1024 256 512 \
  --results-json results.json \
  --checkpoint <checkpoint.pth> \
  --save combined.png
```

**All visualize.py arguments:**

| Argument | Default | Description |
|---|---|---|
| `--mode` | `heatmap` | `heatmap`, `pareto`, or `both` |
| `--num-layers` | `6` | Active layers (heatmap) |
| `--num-heads` | `[8]*L` | Heads per layer (heatmap) |
| `--mlp-dim` | `[1024]*L` | MLP dim per layer (heatmap) |
| `--embed-dim` | `512` | Embed dim (heatmap title) |
| `--results-json` | `None` | JSON from run_search.py (pareto) |
| `--checkpoint` | `None` | Supernet checkpoint for accuracy eval (pareto) |
| `--batch-size` | `256` | Batch size for accuracy evaluation |
| `--save` | `None` | Output image path (shows interactively if omitted) |

---

## Config Format

All scripts use a consistent per-layer config dict:

```python
{
    "embed_dim": 512,          # int — global
    "num_layers": 4,           # int — number of active blocks
    "num_heads": [2, 8, 4, 8], # list[int] — one per active layer
    "mlp_dim":  [256, 1024, 512, 1024],  # list[int] — one per active layer
}
```

> `num_heads` and `mlp_dim` must each have exactly `num_layers` values.

---

## Full Workflow

```bash
# 1. Train
python train_supernet.py --num-epochs 200 --num-workers 8

# 2. Sanity-check max subnet
python eval.py --model-path checkpoint.pth \
  --num-layers 6 --num-heads 8 8 8 8 8 8 --mlp-dim 1024 1024 1024 1024 1024 1024

# 3. Search
python run_search.py --checkpoint checkpoint.pth \
  --mac-constraint 600 --n-subnets 200 \
  --embed-dim-options 512 --num-layers-options 6 \
  --num-heads-options 2 4 8 --mlp-dim-options 256 512 1024

# 4. Visualize
python visualize.py --mode both \
  --num-layers 6 --num-heads 2 8 4 8 2 4 --mlp-dim 256 1024 512 1024 256 512 \
  --results-json results.json --checkpoint checkpoint.pth --save combined.png
```

"""
predictor.py

Phase 3 of the NAS pipeline: trains a small MLP predictor on the dataset
collected by collect_predictor_data.py.

Input features (per sample):
    - architecture config:   embed_dim, num_heads, mlp_dim, num_layers  (4 values)
    - internal signals:      mean/std/max of activation_norm,
                             gradient_norm, attention_entropy            (9 values)
    Total input dim: 13

Output heads:
    - accuracy   (regression)
    - flops      (regression)
    - params     (regression)

All targets are normalized to [0, 1] during training and denormalized
at inference time so predictions are interpretable.

Usage:
    # Train
    python predictor.py --dataset predictor_dataset.pt --output predictor.pt

    # The trained predictor is used by search.py in Phase 4.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(record: dict) -> np.ndarray:
    """
    Flatten a dataset record into a 1D numpy feature vector.

    Config features (4):
        embed_dim, num_heads, mlp_dim, num_layers

    Internal signal features (9):
        activation_norm  (mean, std, max)
        gradient_norm    (mean, std, max)
        attention_entropy(mean, std, max)
    """
    cfg = record["config"]
    config_features = np.array([
        cfg["embed_dim"],
        cfg["num_heads"],
        cfg["mlp_dim"],
        cfg["num_layers"],
    ], dtype=np.float32)

    sig = record["signals"]
    signal_features = np.array([
        sig["activation_norm"]["mean"],
        sig["activation_norm"]["std"],
        sig["activation_norm"]["max"],
        sig["gradient_norm"]["mean"],
        sig["gradient_norm"]["std"],
        sig["gradient_norm"]["max"],
        sig["attention_entropy"]["mean"],
        sig["attention_entropy"]["std"],
        sig["attention_entropy"]["max"],
    ], dtype=np.float32)

    return np.concatenate([config_features, signal_features])  # shape: (13,)


def extract_targets(record: dict) -> np.ndarray:
    """Extract (accuracy, flops, params) as a 1D array."""
    return np.array([
        record["accuracy"],
        record["flops"],
        record["params"],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ArchDataset(Dataset):
    """
    Wraps the list of records saved by collect_predictor_data.py.
    Normalizes features and targets to [0, 1] using min-max scaling.
    """

    def __init__(self, records: list[dict]):
        features = np.stack([extract_features(r) for r in records])  # (N, 13)
        targets  = np.stack([extract_targets(r)  for r in records])  # (N, 3)

        # min-max normalization — store stats for denormalization at inference
        self.feature_min = features.min(axis=0)
        self.feature_max = features.max(axis=0)
        self.target_min  = targets.min(axis=0)
        self.target_max  = targets.max(axis=0)

        # avoid division by zero when all values are identical (e.g. single config)
        feat_range = np.where(self.feature_max - self.feature_min == 0, 1,
                              self.feature_max - self.feature_min)
        tgt_range  = np.where(self.target_max  - self.target_min  == 0, 1,
                              self.target_max  - self.target_min)

        self.features = (features - self.feature_min) / feat_range
        self.targets  = (targets  - self.target_min)  / tgt_range

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.targets  = torch.tensor(self.targets,  dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def denormalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to original scale."""
        tgt_range = np.where(self.target_max - self.target_min == 0, 1,
                             self.target_max - self.target_min)
        scale = torch.tensor(tgt_range, dtype=torch.float32)
        shift = torch.tensor(self.target_min, dtype=torch.float32)
        return targets * scale + shift


# ---------------------------------------------------------------------------
# Predictor MLP
# ---------------------------------------------------------------------------

class PredictorMLP(nn.Module):
    """
    Small MLP: 13 → 128 → 64 → 3

    Three separate output heads (one per target) allow the network to
    learn independent representations for accuracy, FLOPs, and params.
    Outputs are passed through sigmoid so predictions stay in [0, 1]
    (matching the normalized targets).
    """

    def __init__(self, input_dim: int = 13, hidden_dim: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )

        # separate head per output so gradients don't interfere
        self.head_accuracy = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.head_flops    = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.head_params   = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns tensor of shape (B, 3): [accuracy, flops, params]"""
        shared = self.shared(x)
        acc    = self.head_accuracy(shared)
        flops  = self.head_flops(shared)
        params = self.head_params(shared)
        return torch.cat([acc, flops, params], dim=-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_predictor(
    dataset: ArchDataset,
    hidden_dim: int = 128,
    epochs: int = 200,
    lr: float = 1e-3,
    val_split: float = 0.2,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> tuple[PredictorMLP, dict]:

    # train / val split
    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model = PredictorMLP(input_dim=13, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= max(len(val_loader), 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:>3}/{epochs}  "
                  f"train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")

    return model, history


# ---------------------------------------------------------------------------
# Inference helper used by search.py
# ---------------------------------------------------------------------------

class Predictor:
    """
    Thin wrapper around PredictorMLP that handles feature extraction,
    normalization, and denormalization.  This is what search.py imports.

    Usage:
        predictor = Predictor.load("predictor.pt")
        acc, flops, params = predictor.predict(record)
    """

    def __init__(self, model: PredictorMLP, dataset: ArchDataset):
        self.model   = model
        self.dataset = dataset  # carries normalization stats
        self.model.eval()

    def predict(self, record: dict) -> tuple[float, float, float]:
        """
        Given a record dict (must have 'config' and 'signals' keys),
        return (predicted_accuracy, predicted_flops, predicted_params).
        """
        feat_range = np.where(
            self.dataset.feature_max - self.dataset.feature_min == 0, 1,
            self.dataset.feature_max - self.dataset.feature_min
        )
        x = (extract_features(record) - self.dataset.feature_min) / feat_range
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_norm = self.model(x)  # (1, 3) normalized

        pred = self.dataset.denormalize_targets(pred_norm.squeeze(0))
        acc, flops, params = pred.tolist()
        return acc, flops, params

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "feature_min": self.dataset.feature_min,
            "feature_max": self.dataset.feature_max,
            "target_min":  self.dataset.target_min,
            "target_max":  self.dataset.target_max,
        }, path)
        print(f"Predictor saved to {path}")

    @classmethod
    def load(cls, path: str, hidden_dim: int = 128) -> "Predictor":
        ckpt = torch.load(path, map_location="cpu")
        model = PredictorMLP(input_dim=13, hidden_dim=hidden_dim)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # reconstruct a dummy dataset shell just to carry normalization stats
        dummy = object.__new__(ArchDataset)
        dummy.feature_min = ckpt["feature_min"]
        dummy.feature_max = ckpt["feature_max"]
        dummy.target_min  = ckpt["target_min"]
        dummy.target_max  = ckpt["target_max"]

        return cls(model, dummy)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train NAS performance predictor")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to predictor_dataset.pt from collect_predictor_data.py")
    parser.add_argument("--output", type=str, default="predictor.pt",
                        help="Where to save the trained predictor")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int,   default=128)
    parser.add_argument("--batch-size", type=int,   default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    records = torch.load(args.dataset)
    print(f"Loaded {len(records)} records from {args.dataset}")

    dataset = ArchDataset(records)
    print(f"Feature dim: {dataset.features.shape[1]}  "
          f"(4 config + 9 signal features)")

    model, history = train_predictor(
        dataset,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )

    predictor = Predictor(model, dataset)
    predictor.save(args.output)

    # quick sanity check on the first record
    print("\nSanity check on first record:")
    r = records[0]
    acc, flops, params = predictor.predict(r)
    print(f"  True:      acc={r['accuracy']:.4f}  "
          f"flops={r['flops']:,}  params={r['params']:,}")
    print(f"  Predicted: acc={acc:.4f}  "
          f"flops={flops:,.0f}  params={params:,.0f}")

    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
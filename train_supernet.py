import os
from tqdm import tqdm
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

# from timm.data import create_transform
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.optim import Adam
import matplotlib.pyplot as plt

# internal imports
from modules.super_net import SuperNet


# interface for search space
class SearchSpace:
    def __init__(
        self,
        embed_dim_options: list,
        num_heads_options: list,
        mlp_dim_options: list,
        num_layers_options: list,
    ):
        self.embed_dim_options = embed_dim_options
        self.num_heads_options = num_heads_options
        self.mlp_dim_options = mlp_dim_options
        self.num_layers_options = num_layers_options

    def get_max_config(self):
        return {
            "embed_dim": max(self.embed_dim_options),
            "num_heads": max(self.num_heads_options),
            "mlp_dim": max(self.mlp_dim_options),
            "num_layers": max(self.num_layers_options),
        }

    def get_min_config(self):
        return {
            "embed_dim": min(self.embed_dim_options),
            "num_heads": min(self.num_heads_options),
            "mlp_dim": min(self.mlp_dim_options),
            "num_layers": min(self.num_layers_options),
        }

    def sample_random_config(self):
        return {
            "embed_dim": random.choice(self.embed_dim_options),
            "num_heads": random.choice(self.num_heads_options),
            "mlp_dim": random.choice(self.mlp_dim_options),
            "num_layers": random.choice(self.num_layers_options),
        }


def train_one_epoch_sandwich(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    search_space: SearchSpace,
    num_random_subnets=2,
):
    model.train()
    total_loss = 0.0
    train_accuracy = 0.0
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # sandwich rule
        # 1. forward pass with full model and compute loss and backpropagate
        # get max subnet config
        max_config = search_space.get_max_config()
        model.set_active_subnet(max_config)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # only keep track of max loss
        total_loss += loss.item()
        train_accuracy += (
            outputs.argmax(dim=1) == targets
        ).sum().item() / targets.size(0)

        # 2. forward pass with smallest model and compute loss and backpropagate
        min_config = search_space.get_min_config()
        model.set_active_subnet(min_config)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # 3. forward pass with randomly sampled sub-models and compute loss and backpropagate
        for _ in range(num_random_subnets):
            random_config = search_space.sample_random_config()
            if random_config == max_config or random_config == min_config:
                continue  # skip if it matches max or min config
            model.set_active_subnet(random_config)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

        # 4. take one single optimization step for all the above passes
        optimizer.step()

    average_loss = total_loss / len(dataloader)
    average_accuracy = train_accuracy / len(dataloader)
    return average_loss, average_accuracy


def build_dataloader(batch_size=128, validation_split=None):
    # transform = create_transform(
    #     input_size=img_size,
    #     is_training=True,
    #     color_jitter=0.4,
    #     auto_augment='rand-m9-mstd0.5-inc1',
    #     interpolation='bicubic',
    #     re_prob=0.25,
    #     re_mode='pixel',
    #     re_count=1,
    # )
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )

    train_dataset = CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_loader = None
    num_workers = max(0, (os.cpu_count() or 0) - 4)

    if validation_split:
        val_size = int(len(train_dataset) * validation_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_dataset = CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, val_loader


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return average_loss, accuracy


def save_model(model, path):
    torch.save(model.state_dict(), path)


def reload_model(model, path):
    model.load_state_dict(torch.load(path))


def plot_training_curves(train_stats):
    plt.figure(figsize=(10, 5))
    for label, losses in train_stats.items():
        plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Curves")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # search space for NAS; these would be used
    search_space = SearchSpace(
        embed_dim_options=[512],
        num_heads_options=[8],
        mlp_dim_options=[512, 1024],
        num_layers_options=[6],
    )

    max_config = search_space.get_max_config()
    print(f"Max subnet config from search space: {max_config}")
    # get config from json fileTest Loss: {test_loss:.4f},
    config = {
        "img_size": 32,
        "patch_size": 4,
        "embed_dim": max_config["embed_dim"],  # 512,
        "num_layers": max_config["num_layers"],  # 6,
        "num_heads": max_config["num_heads"],  # 8,
        "mlp_dim": max_config["mlp_dim"],
        "num_classes": 10,
        "dropout": 0.1,
        "batch_size": 128,
        "num_epochs": 3,
        "learning_rate": 3e-4,
        "validation_split": 0.1,
        "num_random_subnets": 2,  # number of random subnets to sample for each batch
        "kd_ratio": 0.0,
    }

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = SuperNet(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )

    model.to(device)
    train_loader, test_loader, val_loader = build_dataloader(
        batch_size=config["batch_size"],
        validation_split=config["validation_split"],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    # train the model multiple steps for each design dimension
    # 1. train the full model for one epoch
    # Create an array of design directions to sample fromTest Loss: {test_loss:.4f},
    # for each design dimension, train the model with that specific design direction
    train_stats = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    for epoch in range(config["num_epochs"]):
        train_loss, train_accuracy = train_one_epoch_sandwich(
            model, train_loader, optimizer, criterion, device, search_space
        )
        train_stats["train_loss"].append(train_loss)
        train_stats["train_accuracy"].append(train_accuracy)
        if val_loader is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            train_stats["val_loss"].append(val_loss)
            train_stats["val_accuracy"].append(val_accuracy)
            print(
                f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
            )
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save the final model
    save_model(model, "final_supernet.pth")
    plot_training_curves(
        {"Train Loss": train_stats["train_loss"], "Val Loss": train_stats["val_loss"]}
    )

    # TODO: add functionality to save intermediate checkpoints and reload from them for resuming training or for evaluation.

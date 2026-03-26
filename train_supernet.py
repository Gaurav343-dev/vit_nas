from collections import defaultdict
import os
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from timm import create_model
from timm.utils.model import freeze
from timm.loss.cross_entropy import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

# from timm.data import create_transform
from torch.optim import Adam, AdamW
import matplotlib.pyplot as plt

# internal imports
from modules.super_net import SuperNet
from utils.measurements import get_parameters_size
from utils.data_handler import build_dataloader
from eval import evaluate, evaluate_teacher


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



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

    def set_training_dim(self, key, value):
        if key == "embed_dim":
            self.embed_dim_options = [value]
        elif key == "num_heads":
            self.num_heads_options = [value]
        elif key == "mlp_dim":
            self.mlp_dim_options = [value]
        elif key == "num_layers":
            self.num_layers_options = [value]
        else:
            raise ValueError(f"Invalid key: {key}")


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_224 = F.interpolate(
            inputs, size=(224, 224), mode="bicubic", align_corners=False
        )
        optimizer.zero_grad()
        outputs = model(inputs_224)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return average_loss, accuracy


def train_one_epoch_sandwich(
    model,
    dataloader,
    optimizer,
    criterion,
    kd_criterion: SoftTargetCrossEntropy,
    device,
    search_space: SearchSpace,
    num_random_subnets=2,
    teacher_model=None,
    kd_ratio=0.0,
):
    model.train()
    total_loss = 0.0
    train_accuracy = 0.0

    for inputs, targets in tqdm(dataloader, desc="Sandwich Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        teacher_outputs = None
        # get teacher outputs for distillation if teacher model is provided
        if teacher_model is not None and kd_ratio > 0.0:
            with torch.no_grad():
                inputs_224 = F.interpolate(
                    inputs, size=(224, 224), mode="bicubic", align_corners=False
                )
                teacher_logits = teacher_model(inputs_224)
                # TODO: do you need softmax here ??
                # convert teacher logits to soft labels
                teacher_soft_outputs = F.softmax(teacher_logits, dim=1)
                teacher_outputs = teacher_soft_outputs.detach()  # detach from computation graph

        # sandwich rule
        # 1. forward pass with full model and compute loss and backpropagate
        # get max subnet config
        max_config = search_space.get_max_config()
        model.set_active_subnet(max_config)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if teacher_outputs is not None:
            kd_loss = kd_criterion(outputs, teacher_outputs)
            # nn.KLDivLoss()(
            #     nn.LogSoftmax(dim=1)(outputs / 4),
            #     nn.Softmax(dim=1)(teacher_outputs / 4),
            # ) * (4 * 4)  # temperature scaling
            loss = loss + kd_loss * kd_ratio

        loss.backward()
        max_loss = loss
        # only keep track of max loss
        total_loss += loss.item()
        train_accuracy += (
            outputs.argmax(dim=1) == targets
        ).sum().item() / targets.size(0)

        # 2. forward pass with smallest model and compute loss and backpropagate
        min_config = search_space.get_min_config()
        model.set_active_subnet(min_config)
        outputs = model(inputs)
        min_loss = criterion(outputs, targets)
        if teacher_outputs is not None:
            kd_loss = kd_criterion(outputs, teacher_outputs)
            # nn.KLDivLoss()(
            #     nn.LogSoftmax(dim=1)(outputs / 4),
            #     nn.Softmax(dim=1)(teacher_outputs / 4),
            # ) * (4 * 4)  # temperature scaling
            min_loss = min_loss + kd_loss * kd_ratio
        min_loss.backward()

        # 3. forward pass with randomly sampled sub-models and compute loss and backpropagate
        intermediate_loss = []
        for _ in range(num_random_subnets):
            random_config = search_space.sample_random_config()
            if random_config == max_config:
                intermediate_loss.append(max_loss.item())
                continue
            if random_config == min_config:
                intermediate_loss.append(min_loss.item())
                continue  # skip if it matches max or min config
            model.set_active_subnet(random_config)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if teacher_outputs is not None:
                kd_loss = kd_criterion(outputs, teacher_outputs)
                # nn.KLDivLoss()(
                #     nn.LogSoftmax(dim=1)(outputs / 4),
                #     nn.Softmax(dim=1)(teacher_outputs / 4),
                # ) * (4 * 4)  # temperature scaling
                loss = loss + kd_loss * kd_ratio
            loss.backward()
            intermediate_loss.append(loss.item())

        mean_intermediate_loss = sum(intermediate_loss) / len(intermediate_loss)

        # 4. take one single optimization step for all the above passes
        optimizer.step()

    average_loss = total_loss / len(dataloader)
    average_accuracy = train_accuracy / len(dataloader)
    return average_loss, average_accuracy, min_loss.item(), mean_intermediate_loss


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
    parser = argparse.ArgumentParser(description="Train SuperNet based on ViT architecture")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # search space for NAS; these would be used
    search_space = SearchSpace(
        embed_dim_options=[512],
        num_heads_options=[2, 4, 8],
        mlp_dim_options=[1024],  # [512, 1024],
        num_layers_options=[6],  # [2, 4, 6],
    )

    # set random seed for reproducibility
    set_seed(args.seed)

    max_config = search_space.get_max_config()
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
        "num_epochs": 1,
        "learning_rate": 3e-4,
        "warmup_lr": 1e-6,
        "warmup_epochs": 5,
        "validation_split": 0.1,
        "num_random_subnets": 2,  # number of random subnets to sample for each batch
        "kd_ratio": 0.5,  # weight for knowledge distillation loss (between 0 and 1)
        "teacher_model_path": "teacher_model.pth",
        "teacher_model_name": "vit_small_patch16_224",
        "smoothing": 0.1,  # label smoothing factor for distillation loss
        "num_teacher_epochs": 1,
    }

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load pretrained teacher model
    teacher_model = create_model(
        config["teacher_model_name"],
        pretrained=True,
        num_classes=config["num_classes"],
    )
    # freeze(teacher_model)
    teacher_model.to(device)

    # Fine-tune the teacher model
    if config["kd_ratio"] > 0.0:
        train_loader, test_loader, val_loader = build_dataloader(
            batch_size=config["batch_size"],
            validation_split=config["validation_split"],
            img_size=config["img_size"],
        )
        criterion = nn.CrossEntropyLoss()
        if os.path.exists("teacher_model.pth"):
            print("Loading pretrained teacher model...")
            reload_model(teacher_model, "teacher_model.pth")
            test_loss, test_accuracy = evaluate_teacher(
                teacher_model, test_loader, criterion, device, img_size=224
            )
            print(
                f"Finetuned Teacher {config['teacher_model_name']} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
            )
        else:
            print("Started learning for teacher model...")
            # train the supernet model on the full config
            optimizer = Adam(teacher_model.parameters(), lr=config["learning_rate"])

            train_stats = {
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": [],
            }
            for epoch in range(config["num_teacher_epochs"]):
                train_loss, train_accuracy = train_one_epoch(
                    teacher_model, train_loader, optimizer, criterion, device
                )
                train_stats["train_loss"].append(train_loss)
                train_stats["train_accuracy"].append(train_accuracy)
                if val_loader is not None:
                    val_loss, val_accuracy = evaluate_teacher(
                        teacher_model, val_loader, criterion, device
                    )
                    train_stats["val_loss"].append(val_loss)
                    train_stats["val_accuracy"].append(val_accuracy)
                    print(
                        f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
                    )
            test_loss, test_accuracy = evaluate_teacher(
                teacher_model, test_loader, criterion, device, img_size=224
            )
            print(
                f"Finetuned Teacher {config['teacher_model_name']} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
            )
            save_model(teacher_model, "teacher_model.pth")
            plot_training_curves(
                {
                    "Train Loss": train_stats["train_loss"],
                    "Val Loss": train_stats["val_loss"],
                }
            )
    # after training freeze the teacher model
    freeze(teacher_model)

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
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.05)

    warmup_epochs = config["warmup_epochs"]
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )

    # kd_criterion = SoftTargetCrossEntropy()
    # replace with label smoothing
    kd_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)


    # train the model multiple steps for each design dimension
    # 1. train the full model for one epoch
    # Create an array of design directions to sample fromTest Loss: {test_loss:.4f},
    # for each design dimension, train the model with that specific design direction
    # let's train the model across mlp dimensions
    # train_stats keeps track of losses across different design dimensions for plotting later
    train_stats = defaultdict(list)
    for epoch in range(config["num_epochs"]):
        if epoch < warmup_epochs:
            warmup_lr = config["warmup_lr"]
            base_lr = config["learning_rate"]
            lr_step = (base_lr - warmup_lr) / warmup_epochs
            current_lr = warmup_lr + lr_step * epoch
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        train_loss, train_accuracy, min_loss, mean_intermediate_loss = (
            train_one_epoch_sandwich(
                model,
                train_loader,
                optimizer,
                criterion,
                kd_criterion,
                device,
                search_space,
                teacher_model=teacher_model,
                kd_ratio=config["kd_ratio"],
            )
        )
        train_stats["train_loss"].append(train_loss)
        train_stats["train_accuracy"].append(train_accuracy)
        train_stats["min_loss"].append(min_loss)
        train_stats["mean_intermediate_loss"].append(mean_intermediate_loss)
        if val_loader is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            train_stats["val_loss"].append(val_loss)
            train_stats["val_accuracy"].append(val_accuracy)
            print(
                f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, \
                Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, \
                Val Accuracy: {val_accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        if epoch >= warmup_epochs:
            lr_scheduler.step()

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save the final model
    save_model(model, "final_supernet.pth")
    plot_training_curves(
        {
            "Train Loss": train_stats["train_loss"],
            "Val Loss": train_stats["val_loss"],
            "Min Loss": train_stats["min_loss"],
            "Mean Intermediate Loss": train_stats["mean_intermediate_loss"],
        }
    )

    # plot accuracy curves
    plot_training_curves(
        {
            "Train Accuracy": train_stats["train_accuracy"],
            "Val Accuracy": train_stats["val_accuracy"],
        }
    )

    # TODO: add functionality to save intermediate checkpoints and reload from them for resuming training or for evaluation.

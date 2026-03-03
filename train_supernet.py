import torch
from torch import nn
from torch.utils.data import DataLoader
from timm.data import create_transform
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.optim import Adam
import matplotlib.pyplot as plt

# internal imports
from modules.super_net import SuperNet

def train_one_epoch_sandwich(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        # sandwich rule
        # 1. forward pass with full model and compute loss and backpropagate
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # 2. forward pass with smallest model and compute loss and backpropagate
        # 3. forward pass with randomly sampled sub-models and compute loss and backpropagate
        # 4. take one single optimization step for all the above passes        
        
        optimizer.step()
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    return average_loss

def build_dataloader(batch_size=128, img_size=224):
    transform = create_transform(
        input_size=img_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
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
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curves')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # get config from json file
    config = {
        "img_size": 32,
        "patch_size": 4,
        "embed_dim": 512,
        "num_layers": 4,
        "num_heads": 4,
        "mlp_dim": 1024,
        "num_classes": 10,
        "dropout": 0.1,
        "batch_size": 128,
        "num_epochs": 1,
        "learning_rate": 0.001
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
        dropout=config["dropout"]
    )

    model.to(device)
    train_loader, test_loader = build_dataloader(batch_size=config["batch_size"], img_size=config["img_size"])
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    
    # train the model multiple steps for each design dimension
    # 1. train the full model for one epoch
    # Create an array of design directions to sample from
    # for each design dimension, train the model with that specific design direction 
    train_stats = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": []
    }
    for epoch in range(config["num_epochs"]):
        train_loss = train_one_epoch_sandwich(model, train_loader, optimizer, criterion, device)
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        train_stats["train_loss"].append(train_loss)
        train_stats["test_loss"].append(test_loss)
        train_stats["train_accuracy"].append(accuracy)
        train_stats["test_accuracy"].append(accuracy)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the final model
    save_model(model, "final_supernet.pth")
    plot_training_curves({'Train Loss': train_stats["train_loss"], 'Test Loss': train_stats["test_loss"]})


    # TODO: add functionality to save intermediate checkpoints and reload from them for resuming training or for evaluation.
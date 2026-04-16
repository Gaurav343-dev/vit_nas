import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from timm import create_model

from utils.data_handler import build_dataloader
from modules.super_net import SuperNet

def reload_model(model, path):
    model.load_state_dict(torch.load(path))

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

def evaluate_teacher(model, dataloader, criterion, device, img_size=224):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating Teacher"):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_224 = F.interpolate(
                inputs, size=(img_size, img_size), mode="bicubic", align_corners=False
            )
            outputs = model(inputs_224)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return average_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model_path", type=str, default="final_supernet.pth", help="Path to the trained model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--img_size", type=int, default=32, help="Image size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build testloader
    _, test_loader, _ = build_dataloader(batch_size=args.batch_size, img_size=args.img_size)
    
    criterion = torch.nn.CrossEntropyLoss()

    if os.path.exists(args.model_path):
        model = SuperNet(
            img_size=args.img_size,
            patch_size=4,
            embed_dim=512,
            num_heads=8,
            num_layers=6,
            mlp_dim=1024,
            num_classes=10,
            dropout=0.1,
        )
        # Load the model
        reload_model(model, args.model_path)
        model.to(device)

    # Evaluate
    avg_loss, accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # load pretrained teacher model
    teacher_model = create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=10,
    )

    reload_model(teacher_model, "teacher_model.pth")
    teacher_model.to(device)

    # Evaluate teacher model
    teacher_avg_loss, teacher_accuracy = evaluate_teacher(teacher_model, test_loader, criterion, device, img_size=224)
    print(f"Teacher Evaluation - Loss: {teacher_avg_loss:.4f}, Accuracy: {teacher_accuracy:.4f}")

if __name__ == "__main__":
    main()


# 
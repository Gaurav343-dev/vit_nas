import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Evaluate a supernet subnet on CIFAR-10")

    # supernet architecture (must match the checkpoint)
    parser.add_argument("--model-path",  type=str, default="final_supernet.pth")
    parser.add_argument("--batch-size",  type=int, default=128)
    parser.add_argument("--img-size",    type=int, default=32)
    parser.add_argument("--patch-size",  type=int, default=4)
    parser.add_argument("--embed-dim",   type=int, default=512, help="Supernet max embed dim")
    parser.add_argument("--max-heads",   type=int, default=8,   help="Supernet max num heads")
    parser.add_argument("--max-layers",  type=int, default=6,   help="Supernet max num layers")
    parser.add_argument("--max-mlp-dim", type=int, default=1024,help="Supernet max MLP dim")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--dropout",     type=float, default=0.1)

    # subnet config (per-layer lists; defaults to max subnet)
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Number of active layers in the subnet")
    parser.add_argument("--num-heads",  type=int, nargs="+", default=None,
                        help="Heads per layer, e.g. --num-heads 2 8 4 8")
    parser.add_argument("--mlp-dim",    type=int, nargs="+", default=None,
                        help="MLP dim per layer, e.g. --mlp-dim 256 1024 512 1024")

    # optional teacher evaluation
    parser.add_argument("--eval-teacher", action="store_true",
                        help="Also evaluate the teacher model (requires teacher_model.pth)")
    parser.add_argument("--teacher-model-path", type=str, default="teacher_model.pth")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load supernet
    model = SuperNet(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.max_heads,
        num_layers=args.max_layers,
        mlp_dim=args.max_mlp_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
    )
    reload_model(model, args.model_path)
    model.to(device)

    # resolve subnet config — default to max subnet
    L = args.num_layers if args.num_layers is not None else args.max_layers
    H = args.num_heads  if args.num_heads  is not None else [args.max_heads] * L
    M = args.mlp_dim    if args.mlp_dim    is not None else [args.max_mlp_dim] * L

    if len(H) != L or len(M) != L:
        parser.error(f"--num-heads and --mlp-dim must each have exactly --num-layers={L} values")

    subnet_config = {"embed_dim": args.embed_dim, "num_layers": L, "num_heads": H, "mlp_dim": M}
    print(f"\nSubnet config: {subnet_config}")

    model.set_active_subnet(subnet_config)
    subnet = model.get_active_subnet().to(device)

    total_params = sum(p.numel() for p in subnet.parameters()) / 1e6
    total_macs   = model.get_macs() / 1e6
    print(f"Params: {total_params:.2f}M  |  MACs: {total_macs:.2f}M")

    _, test_loader, _ = build_dataloader(batch_size=args.batch_size, img_size=args.img_size)
    criterion = torch.nn.CrossEntropyLoss()

    avg_loss, accuracy = evaluate(subnet, test_loader, criterion, device)
    print(f"\nTest Loss: {avg_loss:.4f}  |  Test Accuracy: {accuracy*100:.2f}%")

    if args.eval_teacher:
        from timm import create_model as timm_create_model
        teacher_model = timm_create_model("vit_small_patch16_224", pretrained=False, num_classes=args.num_classes)
        reload_model(teacher_model, args.teacher_model_path)
        teacher_model.to(device)
        t_loss, t_acc = evaluate_teacher(teacher_model, test_loader, criterion, device, img_size=224)
        print(f"Teacher — Loss: {t_loss:.4f}  |  Accuracy: {t_acc*100:.2f}%")

if __name__ == "__main__":
    main()
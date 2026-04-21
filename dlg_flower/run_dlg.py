"""Standalone DLG runner — reproduces the official notebook without Flower."""

import argparse
import torch
from dlg_flower.task import LeNet, weights_init, dlg_attack, load_cifar100

def main():
    parser = argparse.ArgumentParser(description="DLG attack on CIFAR-100")
    parser.add_argument("--index", type=int, default=25)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--save-dir", type=str, default="dlg_results")
    args = parser.parse_args()

    torch.manual_seed(50)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on %s" % device)

    dst = load_cifar100()
    net = LeNet().to(device)
    net.apply(weights_init)

    dlg_attack(net=net, dst=dst, img_index=args.index,
               num_iterations=args.iters, save_dir=args.save_dir)

if __name__ == "__main__":
    main()

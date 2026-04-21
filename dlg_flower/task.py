import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchvision import datasets, transforms


# Transforms
tp = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
])
tt = transforms.ToPILImage()


# Helpers 
def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


# inicialização dos pesos
def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


# LeNet 
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1), act(),
        )
        self.fc = nn.Sequential(nn.Linear(768, 100))

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# CIFAR-100
def load_cifar100():
    return datasets.CIFAR100("~/.torch", download=True)


# ataque DLG à partir dos gradientes
def dlg_attack_from_gradients(
    net, original_dy_dx,
    input_shape=(1, 3, 32, 32), num_classes=100,
    num_iterations=300, save_dir="dlg_results", client_id=None,
):
    tag = f"client_{client_id}" if client_id is not None else "victim"
    save_dir = os.path.join(save_dir, tag)
    os.makedirs(save_dir, exist_ok=True)

    device = next(net.parameters()).device
    criterion = cross_entropy_for_onehot

    print(f"\n{'='*60}")
    print(f"  Ataque DLG em {tag} — {num_iterations} iterações L-BFGS")
    print(f"{'='*60}")

    dummy_data = torch.randn(input_shape).to(device).requires_grad_(True)
    dummy_label = torch.randn((1, num_classes)).to(device).requires_grad_(True)
    initial_noise = tt(dummy_data[0].cpu().detach())

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    history, losses = [], []

    for iters in range(num_iterations):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(
                dummy_loss, net.parameters(), create_graph=True
            )
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure()
        losses.append(current_loss.item())
        if iters % 10 == 0:
            print(f"  {tag} | iter {iters:>3d} | loss {current_loss.item():.4f}")
        history.append(tt(dummy_data[0].cpu()))

    recovered_label = torch.argmax(dummy_label, dim=-1).item()
    print(f"  {tag} | Label recuperada: {recovered_label}")

    _save_results(history, losses, initial_noise, None, save_dir,
                  f"Reconstrução DLG — {tag}")
    print(f"  Resultados salvos em: {os.path.abspath(save_dir)}/")
    return dummy_data.detach(), dummy_label.detach()



# Resultados
def _save_results(history, losses, initial_noise, gt_image, save_dir, title):
    os.makedirs(save_dir, exist_ok=True)
    initial_noise.save(os.path.join(save_dir, "iter_0000_noise.png"))
    for i, img in enumerate(history):
        img.save(os.path.join(save_dir, f"iter_{i+1:04d}.png"))

    # 3×10 grid
    fig = plt.figure(figsize=(12, 8))
    for i in range(min(30, len(history) // 10)):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i * 10])
        plt.title("iter=%d" % (i * 10), fontsize=7)
        plt.axis("off")
    fig.savefig(os.path.join(save_dir, "grid_30.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Progressão (4 colunas)
    milestones = [m for m in [0, 1, 10, 20, 50, 100, 150, 200, 250] if m < len(history)]
    if history:
        milestones.append(len(history) - 1)
    all_imgs, all_titles = [], []
    if gt_image:
        all_imgs.append(gt_image)
        all_titles.append("Ground\ntruth")
    all_imgs.append(initial_noise)
    all_titles.append("Iter=0")
    for idx in milestones:
        all_imgs.append(history[idx])
        all_titles.append("Iter=%d" % (idx + 1))
    n = len(all_imgs); cols = 4; rows = (n + cols - 1) // cols
    fig2, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = axes.flatten()
    for i in range(len(axes)):
        if i < n:
            axes[i].imshow(all_imgs[i])
            axes[i].set_title(all_titles[i], fontsize=9)
        axes[i].axis("off")
    fig2.suptitle(title, fontsize=13, y=1.01)
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "progression.png"), bbox_inches="tight", dpi=150)
    plt.close(fig2)

    # Perda
    fig3, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Perda")
    ax.set_title("Convergência DLG")
    if any(l > 0 for l in losses):
        ax.set_yscale("log")
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, "loss_curve.png"), bbox_inches="tight", dpi=150)
    plt.close(fig3)
    if gt_image:
        gt_image.save(os.path.join(save_dir, "ground_truth.png"))

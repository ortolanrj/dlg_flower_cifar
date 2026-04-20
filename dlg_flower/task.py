"""Ataque DLG — porte feito do notebook DLG do artigo Deep Leakage from Gradients.
Autores: Ligeng Zhu, Zhijian Liu, Song Han.

Source: https://gist.github.com/Lyken17/91b81526a8245a028d4f85ccc9191884
"""

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


# Funções helper
def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


# Inicialização dos pesos
def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


# Rede LeNet
# 4 conv layers, 12 filters, Sigmoid, Linear(768, 100)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 100)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# Helper functions Flower
def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)


def load_cifar100():
    dst = datasets.CIFAR100("~/.torch", download=True)
    return dst


# Ataque DLG
def dlg_attack(
    net,
    dst,
    img_index=25,
    num_iterations=300,
    save_dir="dlg_results",
):
    """
    Ataque DLG
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(net.parameters()).device
    criterion = cross_entropy_for_onehot

    # participante honesto
    gt_data = tp(dst[img_index][0]).to(device)
    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label, num_classes=100)

    print("GT label is %d." % gt_label.item(),
          "\nOnehot label is %d." % torch.argmax(gt_onehot_label, dim=-1).item())

    # calcula o gradiente original
    out = net(gt_data)
    y = criterion(out, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    # compartilha os gradientes com outros clientes
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # gera dummy data and label ----
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())

    # salva imagem inicial com ruído
    initial_noise = tt(dummy_data[0].cpu().detach())

    # loop de otimização com L-BFGS
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    history = []
    losses = []
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
            grad_count = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
                grad_count += gx.nelement()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        current_loss = closure()
        losses.append(current_loss.item())
        if iters % 10 == 0:
            print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

    # apresentação dos resultados
    print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())
    print("True label is %d." % gt_label.item())

    # salva imagens individuais
    initial_noise.save(os.path.join(save_dir, "iter_0000_noise.png"))
    for i, img in enumerate(history):
        img.save(os.path.join(save_dir, f"iter_{i+1:04d}.png"))

    fig = plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i * 10])
        plt.title("iter=%d" % (i * 10), fontsize=7)
        plt.axis("off")
    fig.savefig(os.path.join(save_dir, "grid_30.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)

    # imagem de progressão
    milestones = [0, 1, 10, 20, 50, 100, 150, 200, 250, 299]
    n_cols = len(milestones) + 2  # ground truth + initial noise + milestones

    fig2, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 2.8))

    # imagem original
    axes[0].imshow(tt(gt_data[0].cpu()))
    axes[0].set_title("Ground\ntruth", fontsize=8)
    axes[0].axis("off")

    # ruído inicial (true iter=0 before any optimisation)
    axes[1].imshow(initial_noise)
    axes[1].set_title("Iter=0", fontsize=8)
    axes[1].axis("off")

    for j, idx in enumerate(milestones):
        axes[j + 2].imshow(history[idx])
        axes[j + 2].set_title("Iter=%d" % (idx + 1), fontsize=8)
        axes[j + 2].axis("off")

    fig2.suptitle("Progressão da reconstrução via DLG (CIFAR-100)", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "progression.png"),
                 bbox_inches="tight", dpi=150)
    plt.close(fig2)

    # ---- Curva de perda ----
    fig3, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Perda") # Gradient matching loss
    ax.set_title("Convergência do ataque DLG")

    # Usa escala logarítmica se houver somente valores positivos
    positive_losses = [l for l in losses if l > 0]
    if len(positive_losses) > 1:
        ax.set_yscale("log")
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, "loss_curve.png"),
                 bbox_inches="tight", dpi=150)
    plt.close(fig3)

    # ---- Salva imagem original para ter como referência ----
    gt_pil = tt(gt_data[0].cpu())
    gt_pil.save(os.path.join(save_dir, "ground_truth.png"))

    print(f"\nResults saved to: {os.path.abspath(save_dir)}/")
    return dummy_data.detach(), dummy_label.detach()

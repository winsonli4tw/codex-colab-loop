```python
import os, sys, time, math, argparse, csv
from pathlib import Path
import numpy as np
from torch.utils.checkpoint import checkpoint
import psutil, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from torchvision.models import resnet50, ResNet50_Weights
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import lpips
import matplotlib.pyplot as plt
from contextlib import nullcontext
from torchinfo import summary

# ────────── Normalization Definitions ──────────
mean = [0.485, 0.456, 0.406]  # 使用 ImageNet 的 mean
std = [0.229, 0.224, 0.225]   # 使用 ImageNet 的 std

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    def __call__(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

INV = UnNormalize(mean, std)

# ────────── Utility Functions ──────────
def get_cifar10(batch_size, num_workers, data_root):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def mse_kld(recon_x, x, mu, logvar):
    mse = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse, kld

def eval_loss(G, ema_G, val_loader, device):
    G.eval()
    ema_G.apply_shadow()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_onehot = F.one_hot(y, 10).float()
            r, mu, lv = G(x, y_onehot)
            mse, kld = mse_kld(r, x, mu, lv)
            total_loss += (mse + 0.1 * kld).item()
            n += x.size(0)
    ema_G.restore()
    G.train()
    return total_loss / n

def plot_disc_histogram(real_outputs, fake_outputs, epoch, ckpt_dir):
    plt.figure(figsize=(10, 5))
    plt.hist(real_outputs.cpu().numpy(), bins=50, alpha=0.5, label='Real')
    plt.hist(fake_outputs.cpu().numpy(), bins=50, alpha=0.5, label='Fake')
    plt.legend()
    plt.savefig(os.path.join(ckpt_dir, f'disc_hist_ep{epoch:03d}.png'))
    plt.close()

def pretrain_disc(D, train_loader, device, ckpt_dir, epochs=3):
    D.train()
    optD = torch.optim.Adam(D.parameters(), lr=1e-5, betas=(0.0, 0.9))
    for ep in range(epochs):
        for x, _ in tqdm(train_loader, desc=f"Pretraining D Ep {ep+1}/{epochs}"):
            x = x.to(device)
            with (autocast() if torch.cuda.is_available() else nullcontext()):
                d_real = D(x)
                d_loss_real = torch.mean(F.relu(1.0 - d_real))
                fake = torch.randn_like(x)
                d_fake = D(fake)
                d_loss_fake = torch.mean(F.relu(1.0 + d_fake))
                d_loss = d_loss_real + d_loss_fake
            optD.zero_grad()
            d_loss.backward()
            optD.step()
    torch.save(D.state_dict(), os.path.join(ckpt_dir, "D_pretrain.pth"))
    return D

def feature_matching_loss(real_features, fake_features, batch_idx=None, is_last_batch=False):
    loss = 0
    for rf, ff in zip(real_features, fake_features):
        min_bs = min(rf.size(0), ff.size(0))
        loss += F.l1_loss(rf[:min_bs], ff[:min_bs])
        if is_last_batch and rf.size(0) != ff.size(0):
            print(f"[Warning] Last batch feature batch size mismatch: {rf.size(0)} vs {ff.size(0)}")
    return loss / len(real_features)

# ────────── Model Definitions ──────────
class CVAE(nn.Module):
    def __init__(self, z_dim=512, n_cls=10):
        super().__init__()
        enc = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resize = nn.Upsample((224,224), mode='bilinear', align_corners=False)

        self.enc_layers = nn.ModuleList()
        enc_children = list(enc.children())
        self.enc_layers.append(nn.Sequential(*enc_children[:4]))
        self.enc_layers.append(enc_children[4])
        self.enc_layers.append(enc_children[5])
        self.enc_layers.append(enc_children[6])
        self.enc_layers.append(enc_children[7])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(2048+n_cls, z_dim)
        self.fc_lv = nn.Linear(2048+n_cls, z_dim)
        self.dec_fc = nn.Linear(z_dim+n_cls, 512*4*4)

        self.dec_block1 = ResidualBlock(512, 384)
        self.dec

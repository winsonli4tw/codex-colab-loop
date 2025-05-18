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
    # 訓練資料的 transform（可包含增強）
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # 測試資料的 transform（無增強）
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

# ────────── Feature Matching Loss ──────────
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
        self.dec_block2 = ResidualBlock(384, 192)
        self.dec_block2_extra = ResidualBlock(192, 192)
        self.dec_block3 = ResidualBlock(192, 96)
        self.dec_block3_extra = ResidualBlock(96, 96)
        self.dec_block4 = ResidualBlock(96, 48)

        self.final_conv = nn.Sequential(
            nn.Conv2d(48, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, y):
        features = []
        x = self.resize(x)
        for layer in self.enc_layers:
            x = layer(x)
            features.append(x)
        h = self.pool(x).view(x.size(0), -1)
        h = torch.cat([h, y], 1)
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, lv):
        lv = torch.clamp(lv, -5, 5)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def decode(self, z, y):
        h = self.dec_fc(torch.cat([z, y], 1)).view(-1, 512, 4, 4)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.dec_block1(h)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.dec_block2(h)
        h = self.dec_block2_extra(h)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.dec_block3(h)
        h = self.dec_block3_extra(h)
        h = self.dec_block4(h)
        return self.final_conv(h)

    def forward(self, x, y):
        def encode_fn(x, y):
            mu, lv = self.encode(x, y)
            z = self.reparam(mu, lv)
            return mu, lv, z
        mu, lv, z = checkpoint(encode_fn, x, y)
        r = self.decode(z, y)
        return r, mu, lv

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        return F.relu(x)

# SelfAttention 未使用，已註解
# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
#         self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
#         self.value = nn.Conv2d(in_channels, in_channels, 1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         
#     def forward(self, x):
#         batch_size, C, width, height = x.size()
#         query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
#         key = self.key(x).view(batch_size, -1, width * height)
#         energy = torch.bmm(query, key)
#         attention = F.softmax(energy, dim=2)
#         value = self.value(x).view(batch_size, -1, width * height)
#         out = torch.bmm(value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, C, width, height)
#         return self.gamma * out + x

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 96, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(96, 192, 4, 2, 1)),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # self.attention = SelfAttention(192)  # 未使用，已註解
        self.layer2_extra = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(192, 192, 3, 1, 1)),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(192, 384, 4, 2, 1)),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3_extra = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(384, 384, 3, 1, 1)),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(384, 768, 3, 1, 1)),
            nn.BatchNorm2d(768),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(768*4*4, 1))
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        # x = self.attention(x)  # 未使用，已註解
        x = self.layer2_extra(x)
        features.append(x)
        x = self.layer3(x)
        x = self.layer3_extra(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        x = self.classifier(x)
        if return_features:
            return x, features
        return x

def train(args, start_epoch=1):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()  # AMP 相容性
    percept = lpips.LPIPS(net='vgg').to(dev)

    ckpt = Path(args.ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    log_path = ckpt / "training_log.txt"
    log_file = open(log_path, "a")

    def log_message(msg):
        print(msg)
        log_file.write(f"{msg}\n")
        log_file.flush()

    def get_batch_size_for_epoch(start_bs, max_bs, current_epoch, max_epochs):
        target_epoch = int(0.8 * max_epochs)
        if current_epoch >= target_epoch:
            return max_bs
        else:
            ratio = current_epoch / target_epoch
            return int(start_bs + (max_bs - start_bs) * ratio)

    physical_batch_size = 64
    gradient_accumulation_steps = args.batch_size // physical_batch_size

    current_batch_size = args.batch_size // 2
    last_batch_size = current_batch_size
    tr, va = get_cifar10(current_batch_size, min(8, psutil.cpu_count()//2), args.data_root)

    G, D = CVAE(args.z_dim).to(dev), Disc().to(dev)

    # 模型資訊輸出
    print(summary(G, input_size=[(1, 3, 32, 32), (1, 10)]))  # 修正 input_size
    print('#Params G:', sum(p.numel() for p in G.parameters()))
    # Encoder 參數計算（假設為 enc_layers + resize + pool）
    encoder_params = sum(p.numel() for p in G.enc_layers.parameters()) + \
                     sum(param.numel() for layer in [G.resize, G.pool] for param in layer.parameters())
    print('#Params E:', encoder_params)

    ema_G = EMA(G, decay=0.999)
    ema_G.register()

    patience = 20
    best_val_loss = float('inf')
    best_fid = float('inf')
    best_ema_fid = float('inf')
    early_stop_counter = 0
    last_fid = float('inf')
    last_ema_fid = float('inf')
    feat_match_weight = 15.0

    d_pretrain_path = ckpt / "D_pretrain.pth"
    if args.pretrain_d:
        if d_pretrain_path.exists() and not args.force_pretrain:
            log_message(f"載入預訓練判別器權重: {d_pretrain_path}")
            D.load_state_dict(torch.load(d_pretrain_path, map_location=dev))
        else:
            D = pretrain_disc(D, tr, dev, ckpt, epochs=args.pretrain_epochs)

    optG = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)
    optD = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.0, 0.9))
    sched = CosineAnnealingLR(optG, args.epochs, eta_min=1e-8)
    scaler = GradScaler(enabled=use_amp)  # AMP 相容性
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)  # FID 初始化

    training_stats = []

    for ep in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        rec_s = kld_s = d_loss_sum = n = 0
        G.train()
        D.train()

        train_D = ep >= 5  # 前 5 個 epoch 只訓練 G

        current_batch_size = get_batch_size_for_epoch(64, args.batch_size, ep, args.epochs)
        if current_batch_size != last_batch_size:
            tr, va = get_cifar10(current_batch_size, min(8, psutil.cpu_count()//2), args.data_root)
            last_batch_size = current_batch_size
            log_message(f"Updated batch size to {current_batch_size} at epoch {ep}")

        pbar = tqdm(tr, ncols=80)
        d_real_outputs = []
        d_fake_outputs = []
        d_real_correct = 0
        d_fake_correct = 0
        total_samples = 0
        d_real_mean = d_fake_mean = d_recon_mean = 0

        kl_scale = args.kl_max * min(ep / args.kl_warmup, 1.0)  # KL weight 佈局

        if ep < args.adv_warmup_start:
            current_adv_w = 0.0
        elif ep < args.adv_warmup_end:
            progress = (ep - args.adv_warmup_start) / (args.adv_warmup_end - args.adv_warmup_start)
            current_adv_w = args.adv_w * progress
        else:
            warmup_complete_progress = min(1.0, (ep - args.adv_warmup_end) / (args.epochs - args.adv_warmup_end))
            current_adv_w = args.adv_w * (1.0 + 0.2 * math.cos(math.pi * warmup_complete_progress))

        current_adv_w = max(0.1, min(10.0, current_adv_w))

        iter_loader = iter(tr)
        for i in range(len(tr)):
            try:
                x, y = next(iter_loader)
            except StopIteration:
                iter_loader = iter(tr)
                x, y = next(iter_loader)
            x, y = x.to(dev), y.to(dev)
            y1 = F.one_hot(y, 10).float()
            batch_size = x.size(0)

            # Generator Training
            for p in G.parameters():
                p.requires_grad_(True)  # 使用 requires_grad_ 代替直接賦值
            for p in D.parameters():
                p.requires_grad_(False)

            optG.zero_grad()
            for accum_step in range(gradient_accumulation_steps):
                if accum_step < gradient_accumulation_steps - 1:
                    try:
                        x_accum, y_accum = next(iter_loader)
                    except StopIteration:
                        iter_loader = iter(tr)
                        x_accum, y_accum = next(iter_loader)
                    x_accum, y_accum = x_accum.to(dev), y_accum.to(dev)
                    y1_accum = F.one_hot(y_accum, 10).float()
                else:
                    x_accum, y_accum, y1_accum = x, y, y1

                with (autocast() if use_amp else nullcontext()):  # AMP 相容性
                    r, mu, lv = G(x_accum, y1_accum)
                    rec, kld = mse_kld(r, x_accum, mu, lv)

                    if current_adv_w > 0 and train_D:
                        real_output, real_features = D(x_accum, return_features=True)
                        fake_output, fake_features = D(r, return_features=True)
                        adv = -torch.mean(fake_output)
                        fm_loss = feature_matching_loss(real_features, fake_features)

                        with torch.no_grad():
                            d_recon_mean = torch.sigmoid(fake_output).mean().item()

                        if ep > args.adv_warmup_end:
                            z_rand = torch.randn(batch_size, args.z_dim, device=dev)
                            rand_labels = torch.randint(0, 10, (batch_size,), device=dev)
                            rand_onehot = F.one_hot(rand_labels, 10).float()
                            fake_imgs = G.decode(z_rand, rand_onehot)
                            g_fake_rand_output, fake_rand_features = D(fake_imgs, return_features=True)
                            adv_rand = -torch.mean(g_fake_rand_output)
                            fm_loss_rand = feature_matching_loss(real_features, fake_rand_features)
                            adv = 0.5 * (adv + adv_rand)
                            fm_loss = 0.5 * (fm_loss + fm_loss_rand)
                    else:
                        adv = torch.tensor(0.0, device=dev)
                        fm_loss = torch.tensor(0.0, device=dev)

                    if ep > args.kl_warmup:
                        kl_weight = max(0.1, 1.0 - 0.5 * (ep - args.kl_warmup) / (args.epochs - args.kl_warmup))
                    else:
                        kl_weight = kl_scale

                    lG_step = (rec + kl_weight * kld + current_adv_w * adv + feat_match_weight * fm_loss) / batch_size
                    lG_step = lG_step / gradient_accumulation_steps

                scaler.scale(lG_step).backward()

            scaler.step(optG)
            scaler.update()
            ema_G.update()

            rec_s += rec.item()
            kld_s += kld.item()
            n += batch_size

            # Discriminator Training
            if train_D:
                for p in G.parameters():
                    p.requires_grad_(False)  # 使用 requires_grad_ 代替直接賦值
                for p in D.parameters():
                    p.requires_grad_(True)

                optD.zero_grad()
                with (autocast() if use_amp else nullcontext()):  # AMP 相容性
                    with torch.no_grad():
                        r, mu, lv = G(x, y1)
                        # diff_augment 已移除，這裡假設無增強
                        x_aug = x
                        r_aug = r.detach()
                        z_rand = torch.randn(batch_size, args.z_dim, device=dev)
                        fake_rand = G.decode(z_rand, y1)
                        fake_rand_aug = fake_rand.detach()

                        real_label = torch.empty((batch_size,), device=dev).uniform_(0.7, 1.2)
                        fake_label = torch.empty((batch_size,), device=dev).uniform_(0.0, 0.3)

                        d_real = D(x_aug)
                        d_loss_real = F.binary_cross_entropy_with_logits(d_real, real_label)

                        d_fake = D(r_aug)
                        d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_label)

                        d_fake_rand = D(fake_rand_aug)
                        d_loss_fake_rand = F.binary_cross_entropy_with_logits(d_fake_rand, fake_label)

                        d_loss = d_loss_real + 0.5 * (d_loss_fake + d_loss_fake_rand)

                scaler.scale(d_loss).backward()
                scaler.step(optD)
                scaler.update()

                if ep > args.adv_warmup_end or (ep > args.adv_warmup_start and i % 3 == 0):
                    with (autocast() if use_amp else nullcontext()):
                        grad_penalty = compute_gradient_penalty(D, x, r.detach(), dev)
                        grad_penalty_weight = 10.0
                    scaler.scale(grad_penalty_weight * grad_penalty).backward()
                else:
                    grad_penalty = torch.tensor(0.0, device=dev)
                    grad_penalty_weight = 0.0

                d_loss_sum += d_loss.item()

                with torch.no_grad():
                    d_real_mean = torch.sigmoid(d_real).mean().item()
                    d_fake_mean = torch.sigmoid(d_fake).mean().item()
                    d_real_outputs.append(d_real.detach())
                    d_fake_outputs.append(d_fake.detach())
                    d_real_correct += (d_real > 0).float().sum().item()
                    d_fake_correct += (d_fake < 0).float().sum().item()
                    total_samples += batch_size

            pbar.set_description(f"Ep{ep} G:{lG_step.item():.2f} D:{d_loss.item() if train_D else 0:.2f} adv_w:{current_adv_w:.3f} fm_w:{feat_match_weight:.2f}")

            if i % 50 == 0:
                log_message(f"Ep{ep} Batch {i} - G_loss: {lG_step.item():.4f}, D_loss: {d_loss.item() if train_D else 0:.4f}, "
                           f"D(real): {d_real_mean:.4f}, D(fake): {d_fake_mean:.4f}, D(recon): {d_recon_mean:.4f}, "
                           f"Rec: {rec.item()/batch_size:.4f}, KL: {kld.item()/batch_size:.4f}, "
                           f"Adv: {adv.item():.4f}, FM: {fm_loss.item():.4f}, adv_w: {current_adv_w:.4f}, fm_w: {feat_match_weight:.2f}")

        sched.step()

        d_real_acc = d_real_correct / total_samples if train_D else 0
        d_fake_acc = d_fake_correct / total_samples if train_D else 0

        if train_D and d_real_acc > 0.9 and d_fake_acc > 0.9:
            current_adv_w = min(10.0, current_adv_w * 1.1)
            log_message(f"Discriminator too strong, increased adv_w to {current_adv_w:.4f}")
        elif train_D and (d_real_acc < 0.6 or d_fake_acc < 0.6):
            current_adv_w = max(0.1, current_adv_w * 0.9)
            log_message(f"Discriminator too weak, decreased adv_w to {current_adv_w:.4f}")

        tr_l = (rec_s + kld_s) / n
        va_l = eval_loss(G, ema_G, va, dev)
        avg_d_loss = d_loss_sum / len(tr) if train_D else 0
        t_s = time.time() - t0

        if ep % 10 == 0 or ep == 1:
            if train_D and len(d_real_outputs) > 0 and len(d_fake_outputs) > 0:
                all_real = torch.cat(d_real_outputs, dim=0)
                all_fake = torch.cat(d_fake_outputs, dim=0)
                plot_disc_histogram(all_real, all_fake, ep, ckpt)

        fid_val = ''
        ema_fid_val = ''
        if ep % 5 == 0:
            fid.reset()
            G.eval()
            with torch.no_grad():
                for x, _ in va:
                    fid.update(x.to(dev).float(), real=True)  # FID 真實資料
                batch = 100
                for cls in range(10):
                    loops = args.n_gen_per_cls // batch
                    for _ in range(loops):
                        z = torch.randn(batch, args.z_dim, device=dev)
                        y = torch.full((batch,), cls, device=dev)
                        y1 = F.one_hot(y, 10).float()
                        fake = G.decode(z, y1).clamp(0, 1)
                        fid.update(fake, real=False)  # FID 生成資料
            fid_val = fid.compute().item()
            log_message(f"Main FID@Ep{ep}: {fid_val:.2f}")

            fid.reset()
            ema_G.apply_shadow()
            with torch.no_grad():
                for x, _ in va:
                    fid.update(x.to(dev).float(), real=True)
                batch = 100
                for cls in range(10):
                    loops = args.n_gen_per_cls // batch
                    for _ in range(loops):
                        z = torch.randn(batch, args.z_dim, device=dev)
                        y = torch.full((batch,), cls, device=dev)
                        y1 = F.one_hot(y, 10).float()
                        fake = G.decode(z, y1).clamp(0, 1)
                        fid.update(fake, real=False)
            ema_fid_val = fid.compute().item()
            ema_G.restore()
            log_message(f"EMA FID@Ep{ep}: {ema_fid_val:.2f}")

            if ep % 5 == 0:  # Qualitative 影像輸出
                generate_grid(ep, G, args.z_dim, dev, args.ckpt_dir)

            if ep > 5:
                if ema_fid_val > fid_val + 5.0:
                    new_decay = min(0.9999, ema_G.decay + 0.0005)
                    ema_G.decay = new_decay
                    log_message(f"EMA FID worse, increased EMA decay to {new_decay:.6f}")
                elif ema_fid_val < fid_val - 5.0:
                    new_decay = max(0.995, ema_G.decay - 0.0005)
                    ema_G.decay = new_decay
                    log_message(f"EMA FID better, decreased EMA decay to {new_decay:.6f}")

                if fid_val > last_fid:
                    feat_match_weight = min(30.0, feat_match_weight * 1.2)
                    log_message(f"FID increased, increased feat_match_weight to {feat_match_weight:.2f}")
                else:
                    feat_match_weight = max(5.0, feat_match_weight * 0.9)
                    log_message(f"FID decreased, decreased feat_match_weight to {feat_match_weight:.2f}")
                last_fid = fid_val
                last_ema_fid = ema_fid_val

        if isinstance(fid_val, float):
            if fid_val < best_fid:
                best_fid = fid_val
                early_stop_counter = 0
                log_message(f"New best main FID: {fid_val:.2f}")
            if isinstance(ema_fid_val, float) and ema_fid_val < best_ema_fid:
                best_ema_fid = ema_fid_val
                log_message(f"New best EMA FID: {ema_fid_val:.2f}")
            if fid_val >= best_fid:
                early_stop_counter += 1
                log_message(f"FID no improvement, counter: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                log_message(f"\n>>> Early Stopping: {patience} epochs FID no progress <<<")
                break

        training_stats.append({
            "epoch": ep,
            "train_loss": tr_l,
            "val_loss": va_l,
            "d_loss": avg_d_loss,
            "d_real_acc": d_real_acc,
            "d_fake_acc": d_fake_acc,
            "fid": fid_val,
            "ema_fid": ema_fid_val,
            "time": t_s,
            "adv_weight": current_adv_w,
            "feat_match_weight": feat_match_weight
        })

        try:
            with open(ckpt / "metrics.csv", "a", newline="") as f:
                csv.writer(f).writerow([ep,
                                        f"{tr_l:.6f}",
                                        f"{va_l:.6f}",
                                        f"{avg_d_loss:.6f}",
                                        f"{d_real_acc:.6f}",
                                        f"{d_fake_acc:.6f}",
                                        fid_val,
                                        ema_fid_val,
                                        f"{t_s:.1f}"])
        except FileNotFoundError:
            ckpt.mkdir(parents=True, exist_ok=True)
            with open(ckpt / "metrics.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "d_loss", "d_real_acc", "d_fake_acc", "fid", "ema_fid", "time_sec"])
                writer.writerow([ep,
                                f"{tr_l:.6f}",
                                f"{va_l:.6f}",
                                f"{avg_d_loss:.6f}",
                                f"{d_real_acc:.6f}",
                                f"{d_fake_acc:.6f}",
                                fid_val,
                                ema_fid_val,
                                f"{t_s:.1f}"])

        with open(ckpt / "latest_stats.txt", "w") as f:
            f.write(f"Epoch: {ep}/{args.epochs}\n")
            f.write(f"Train Loss: {tr_l:.6f}\n")
            f.write(f"Val Loss: {va_l:.6f}\n")
            f.write(f"D Loss: {avg_d_loss:.6f}\n")
            f.write(f"D Real Accuracy: {d_real_acc:.6f}\n")
            f.write(f"D Fake Accuracy: {d_fake_acc:.6f}\n")
            f.write(f"FID: {fid_val}\n")
            f.write(f"EMA FID: {ema_fid_val}\n")
            f.write(f"Time: {t_s:.1f} sec\n")
            f.write(f"Adversarial Weight: {current_adv_w:.6f}\n")
            f.write(f"Feature Match Weight: {feat_match_weight:.6f}\n")
            f.write(f"KL Scale: {kl_scale:.6f}\n")
            f.write(f"EMA Decay: {ema_G.decay:.6f}\n")

        if ep % args.save_every == 0 or ep == args.epochs:
            torch.save({
                "epoch": ep,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": args.best_val,
                "training_stats": training_stats,
                "fid": best_fid  # 加入 fid
            }, ckpt / "last.pth")

            torch.random.manual_seed(ep)
            n_samples = 64
            z = torch.randn(n_samples, args.z_dim, device=dev)
            base = torch.arange(10, device=dev).repeat_interleave(6)
            extra = torch.randint(0, 10, (n_samples - base.size(0),), device=dev)
            y_labels = torch.cat([base, extra])
            y_onehot = F.one_hot(y_labels, 10).float()
            ema_G.apply_shadow()
            with torch.no_grad():
                samples = G.decode(z, y_onehot)
                grid = make_grid(samples, nrow=8, normalize=True, value_range=(0, 1))
                save_image(grid, ckpt / f"samples_ep{ep:03d}.png")
            ema_G.restore()

            if va_l < args.best_val:
                args.best_val = va_l
                torch.save({
                    "G": G.state_dict(),
                    "epoch": ep,
                    "val_loss": va_l,
                    "fid": fid_val,
                    "ema_fid": ema_fid_val
                }, ckpt / "best.pth")
                log_message(f"New best model! Val Loss: {va_l:.4f}")

                log_message(f"Ep{ep}/{args.epochs} ▶ train {tr_l:.4f} val {va_l:.4f} D_loss {avg_d_loss:.4f} "
                           f"D_real_acc {d_real_acc:.4f} D_fake_acc {d_fake_acc:.4f} | adv_w {current_adv_w:.4f} | fm_w {feat_match_weight:.2f} | {t_s:.1f}s")

# Qualitative 影像輸出
@torch.no_grad()
def generate_grid(epoch, G, z_dim, device, ckpt_dir):
    z = torch.randn(100, z_dim, device=device)
    y = torch.arange(10, device=device).repeat_interleave(10)
    y1 = F.one_hot(y, 10).float()
    fake = G.decode(z, y1).clamp(0, 1)
    grid = make_grid(fake, nrow=10)
    save_image(grid, os.path.join(ckpt_dir, f"samples/epoch_{epoch:03d}.png"))

def save_final_samples(G, z_dim, device, ckpt_dir):
    G.eval()
    samples = []
    with torch.no_grad():
        for cls in range(10):
            z = torch.randn(10, z_dim, device=device)
            y = torch.full((10,), cls, device=device, dtype=torch.long)
            f = G.decode(z, F.one_hot(y, 10).float())
            f = f.clamp(0, 1)
            samples.append(f.cpu())
    all_samples = torch.cat(samples, dim=0)
    grid = make_grid(all_samples, nrow=10, normalize=True, value_range=(0, 1))
    save_image(grid, os.path.join(ckpt_dir, "final_cifar10_generated.png"))

if __name__ == "__main__":
    os.makedirs('samples', exist_ok=True)  # 建立 samples 目錄

    parser = argparse.ArgumentParser(description="Improved CVAE-GAN CIFAR-10 Trainer")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d_lr", type=float, default=1e-6)
    parser.add_argument("--kl_warmup", type=int, default=30)  # 新增旗標
    parser.add_argument("--adv_warmup_start", type=int, default=10)
    parser.add_argument("--adv_warmup_end", type=int, default=40)
    parser.add_argument("--adv_w", type=float, default=2.0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrain_d", dest="pretrain_d", action="store_true")
    parser.add_argument("--no-pretrain_d", dest="pretrain_d", action="store_false")
    parser.set_defaults(pretrain_d=True)
    parser.add_argument("--force_pretrain", action="store_true", default=False)
    parser.add_argument("--pretrain_epochs", type=int, default=3)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    parser.add_argument("--best_val", type=float, default=float('inf'))
    parser.add_argument('--kl_max', type=float, default=0.1)  # 新增旗標
    parser.add_argument('--n_gen_per_cls', type=int, default=1000)  # 新增旗標
    args = parser.parse_args()

    if args.resume and (Path(args.ckpt_dir) / "last.pth").exists():
        ckpt = torch.load(Path(args.ckpt_dir) / "last.pth", map_location='cpu')
        args.best_val = ckpt.get("best_val", float('inf'))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"▶ Resuming from epoch {start_epoch}")
    else:
        start_epoch = 1
        print("▶ Starting training from scratch")

    train(args, start_epoch=start_epoch)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    last_ckpt = torch.load(Path(args.ckpt_dir) / "last.pth", map_location=dev)
    G = CVAE(args.z_dim).to(dev)
    G.load_state_dict(last_ckpt["G"])
    save_final_samples(G, args.z_dim, dev, args.ckpt_dir)

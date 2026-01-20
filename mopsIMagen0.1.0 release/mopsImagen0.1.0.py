import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from tqdm import tqdm

# --- 1. Настройки ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 50  # Пока оставим 50
IMG_DIR = "generated_animals_v2"  # Новая папка

if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)

# --- 2. Подготовка данных (CIFAR-10) ---
transform = transforms.Compose([
    transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# --- 3. Архитектура нейросетей (без изменений) ---
class Generator(nn.Module):
    # ... (код класса Generator остается прежним) ...
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x): return self.main(x)


class Discriminator(nn.Module):
    # ... (код класса Discriminator остается прежним) ...
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )

    def forward(self, x): return self.main(x).view(-1, 1)


netG = Generator().to(DEVICE)
netD = Discriminator().to(DEVICE)
# ИСПРАВЛЕНИЕ: Дискриминатор теперь учится медленнее (LR 0.0001 вместо 0.0002)
optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# --- 4. Обучение с Hot Bar (сглаживание меток) ---
print(f"Начинаю обучение генератора животных v0.2.0 на {torch.cuda.get_device_name(0)}")

for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Эпоха {epoch + 1}/{EPOCHS}")
    for i, (real_imgs, _) in enumerate(pbar):
        b_size = real_imgs.size(0)
        real_imgs = real_imgs.to(DEVICE)

        # Обучаем Дискриминатор
        netD.zero_grad()
        # ИСПРАВЛЕНИЕ: Сглаживание реальных меток (не 1.0, а 0.9)
        label_real = torch.full((b_size, 1), 0.9, device=DEVICE)
        output = netD(real_imgs)
        lossD_real = criterion(output, label_real)

        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake_imgs = netG(noise)
        label_fake = torch.full((b_size, 1), 0.0, device=DEVICE)
        output = netD(fake_imgs.detach())
        lossD_fake = criterion(output, label_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # Обучаем Генератор
        netG.zero_grad()
        output = netD(fake_imgs)
        # Генератор все еще видит цель как 1.0
        lossG = criterion(output, torch.full((b_size, 1), 1.0, device=DEVICE))
        lossG.backward()
        optimizerG.step()

        pbar.set_postfix(D_loss=f"{lossD.item():.4f}", G_loss=f"{lossG.item():.4f}")

    save_image(fake_imgs.data[:25], f"{IMG_DIR}/epoch_{epoch + 1}.png", nrow=5, normalize=True)

print("Готово! Проверь папку generated_animals_v2")


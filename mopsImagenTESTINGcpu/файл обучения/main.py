import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


# --- Архитектура Генератора ---
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), *IMG_SHAPE)


# --- Архитектура Дискриминатора ---
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))


# --- Основные параметры ---
LATENT_DIM = 100  # Размерность вектора шума
IMG_SHAPE = (1, 28, 28)  # Форма изображения (1 канал, 28x28 пикселей)
BATCH_SIZE = 64
LR = 0.0002
NUM_EPOCHS = 15  # Количество проходов обучения (увеличьте для лучшего результата)

# Инициализация моделей, оптимизаторов и функции потерь
generator = Generator(LATENT_DIM, IMG_SHAPE)
discriminator = Discriminator(IMG_SHAPE)
optimizer_G = optim.Adam(generator.parameters(), lr=LR)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR)
criterion = nn.BCELoss()

# Загрузка данных MNIST
os.makedirs("data/mnist", exist_ok=True)
dataloader = DataLoader(
    datasets.MNIST("data/mnist", train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True
)


# --- Функция обучения ---
def train(generator, discriminator, dataloader, num_epochs):
    print("Начало обучения GAN на MNIST...")
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            valid = torch.ones(imgs.size(0), 1)
            fake = torch.zeros(imgs.size(0), 1)

            # --- Тренировка Генератора ---
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), LATENT_DIM)
            gen_imgs = generator(z)
            g_loss = criterion(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # --- Тренировка Дискриминатора ---
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(
                    f"Эпоха [{epoch + 1}/{num_epochs}] Пакет {i}/{len(dataloader)} | Потери D: {d_loss.item():.4f}, Потери G: {g_loss.item():.4f}")

        # Сохраняем пример сгенерированных изображений после каждой эпохи
        # Файлы будут сохранены как mnist_epoch_1.png, mnist_epoch_2.png и т.д.
        save_image(gen_imgs.data[:25], f"mnist_epoch_{epoch + 1}.png", nrow=5, normalize=True)
        print(f"Сохранены образцы после эпохи {epoch + 1}")


# Запуск обучения
train(generator, discriminator, dataloader, NUM_EPOCHS)
torch.save(generator.state_dict(), "generator_final.pth")

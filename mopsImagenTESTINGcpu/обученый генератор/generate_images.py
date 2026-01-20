import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

# !!! ВАЖНО: Класс Generator должен быть точно таким же, как при обучении !!!
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
        # IMG_SHAPE должен быть определен глобально или передан как аргумент
        img_shape = (1, 28, 28)
        return self.model(z).view(z.size(0), *img_shape)

# --- Параметры ---
LATENT_DIM = 100
IMG_SHAPE = (1, 28, 28)
MODEL_PATH = "generator_final.pth" # Имя вашего файла
OUTPUT_FILE = "new_generated_digits.png"
NUM_IMAGES_TO_GENERATE = 10 # Сколько картинок сгенерировать

# 1. Инициализируем пустую модель Генератора
generator = Generator(LATENT_DIM, IMG_SHAPE)

# 2. Загружаем сохраненные веса из файла
if os.path.exists(MODEL_PATH):
    generator.load_state_dict(torch.load(MODEL_PATH))
    generator.eval() # Переводим модель в режим оценки (Evaluation mode)
    print(f"Модель успешно загружена из {MODEL_PATH}")
else:
    print(f"Ошибка: файл {MODEL_PATH} не найден. Сначала обучите модель.")
    exit()

# 3. Генерируем новые изображения
with torch.no_grad(): # Отключаем расчет градиентов, так как мы не обучаем
    # Создаем новые случайные шумы
    z = torch.randn(NUM_IMAGES_TO_GENERATE, LATENT_DIM)
    # Генерируем изображения
    generated_images = generator(z)

# 4. Сохраняем результаты в новый файл
output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
save_image(generated_images, output_path, nrow=NUM_IMAGES_TO_GENERATE, normalize=True)

print(f"Новые {NUM_IMAGES_TO_GENERATE} изображений сохранены в {output_path}")


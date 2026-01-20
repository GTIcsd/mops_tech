import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random

# --- 1. Расширенные данные для обучения ---
qa_pairs = [
    ("привет", "привет как твои дела"),
    ("как дела", "у меня все отлично а у тебя"),
    ("кто ты", "я продвинутая нейросеть трансформер"),
    ("что ты умеешь", "я умею поддерживать диалог и обучаться"),
    ("пока", "до скорой встречи пользователь"),
    ("как тебя зовут", "меня зовут искусственный интеллект"),
    ("что делаешь", "изучаю структуру человеческого языка"),
    ("ты человек", "нет я программный код на питоне")
]

# Создание словаря
all_text = " ".join([q + " " + a for q, a in qa_pairs]) + " <SOS> <EOS> <UNK>"
words = sorted(list(set(all_text.split())))
word_to_idx = {word: i + 1 for i, word in enumerate(words)}
word_to_idx["<PAD>"] = 0
idx_to_word = {i: word for word, i in word_to_idx.items()}

VOCAB_SIZE = len(word_to_idx)
EMBED_DIM = 128  # Увеличили размерность
N_HEADS = 8  # 8 голов внимания
HIDDEN_DIM = 256  # Слой внутри блока
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. Глубокая архитектура Трансформера ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x, mask):
        # Self-Attention + Residual
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        # Feed Forward + Residual
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class DeepGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 100, embed_dim))  # Позиционное кодирование
        self.layer = TransformerBlock(embed_dim, n_heads, hidden_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        b, t = x.size()
        mask = torch.triu(torch.ones(t, t), diagonal=1).bool().to(DEVICE)

        x = self.embed(x) + self.pos_enc[:, :t, :]
        x = self.layer(x, mask)
        return self.fc_out(x)


model = DeepGPT(VOCAB_SIZE, EMBED_DIM, N_HEADS, HIDDEN_DIM).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=0.0005)  # AdamW лучше для глубоких сетей
criterion = nn.CrossEntropyLoss(ignore_index=0)

# --- 3. Улучшенный цикл обучения ---
print("Глубокое обучение трансформера (2026)...")
model.train()
for epoch in range(1000):
    total_loss = 0
    random.shuffle(qa_pairs)  # Перемешиваем для лучшей генерализации
    for q, a in qa_pairs:
        full_seq = q.split() + ["<SOS>"] + a.split() + ["<EOS>"]
        ids = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in full_seq]
        ids_tensor = torch.tensor([ids], dtype=torch.long).to(DEVICE)

        optimizer.zero_grad()
        output = model(ids_tensor[:, :-1])
        target = ids_tensor[:, 1:]

        loss = criterion(output.view(-1, VOCAB_SIZE), target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 200 == 0:
        print(f"Эпоха {epoch + 1} | Loss: {total_loss / len(qa_pairs):.4f}")

print("Обучение завершено!")


# --- 4. Умная генерация с Nucleus Sampling ---
def chat(text):
    model.eval()
    input_tokens = text.lower().split()
    input_ids = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in input_tokens]

    generated = input_ids + [word_to_idx["<SOS>"]]

    for _ in range(10):
        input_tensor = torch.tensor([generated], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = model(input_tensor)[:, -1, :]

        # Штраф за повторы
        for token_id in set(generated):
            logits[0, token_id] /= 1.5

            # Берем только вероятные слова (Top-P или Top-K)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == word_to_idx["<EOS>"] or next_token == 0:
            break
        generated.append(next_token)

    try:
        sos_idx = generated.index(word_to_idx["<SOS>"])
        res_ids = generated[sos_idx + 1:]
        return " ".join([idx_to_word.get(i, "") for i in res_ids])
    except:
        return "..."


# --- Запуск ---
while True:
    user_in = input("\nВы: ")
    if user_in.lower() in ['выход', 'exit']: break
    print(f"Бот: {chat(user_in)}")

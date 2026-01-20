import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import pickle
from tqdm import tqdm

# --- 1. Настройки ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 256
N_HEADS = 8
HIDDEN_DIM = 512
N_LAYERS = 4  # 4 слоя для лучшего понимания смысла
BATCH_SIZE = 64
EPOCHS = 500  # 500 эпох сделают ответы точными
MODEL_FILE = "model_v2.pth"
VOCAB_FILE = "vocab_v2.pkl"


# --- 2. Архитектура Глубокого Трансформера ---
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
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ff(x))
        return x


class DeepModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, hidden_dim, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, embed_dim))
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, n_heads, hidden_dim) for _ in range(n_layers)])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        b, t = x.size()
        mask = torch.triu(torch.ones(t, t, device=DEVICE), diagonal=1).bool()
        x = self.embed(x) + self.pos_enc[:, :t, :]
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)


# --- 3. Подготовка данных ---
def load_data():
    qa_pairs = [
        ("привет", "привет! я puppyGPT 0.1.0"),
        ("как тебя зовут", "меня зовут puppyGPT 0.1.0"),
        ("назови свое имя", "моё имя puppyGPT"),
        ("кто ты", "я твоя личная llm модель"),
        ("как дела", "у меня всё отлично, я учусь!")
    ]
    files = ["aphoristic_facts.dat", "profile_facts_1.dat", "shared_facts.dat", "shared_facts_wiki.dat"]
    for filename in files:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for i in range(len(lines) - 1):
                    q, a = lines[i].strip().lower(), lines[i + 1].strip().lower()
                    if q and a and not q.startswith('#'):
                        qa_pairs.append((q, a))
    return qa_pairs


def create_vocab(qa_pairs):
    all_text = " ".join([q + " " + a for q, a in qa_pairs])
    words = sorted(list(set(all_text.split())))
    w2i = {word: i + 4 for i, word in enumerate(words)}
    w2i.update({"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3})
    i2w = {i: word for word, i in w2i.items()}
    return w2i, i2w


qa_pairs = load_data()

# Загрузка или обучение
if os.path.exists(MODEL_FILE) and os.path.exists(VOCAB_FILE):
    print("--- Найдено сохранение. Загружаю модель... ---")
    with open(VOCAB_FILE, "rb") as f:
        v_data = pickle.load(f)
        word_to_idx, idx_to_word = v_data["word_to_idx"], v_data["idx_to_word"]
    model = DeepModel(len(word_to_idx), EMBED_DIM, N_HEADS, HIDDEN_DIM, N_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()
else:
    print(f"--- Обучаю ГЛУБОКУЮ модель на {DEVICE}... ---")
    word_to_idx, idx_to_word = create_vocab(qa_pairs)
    VOCAB_SIZE = len(word_to_idx)
    model = DeepModel(VOCAB_SIZE, EMBED_DIM, N_HEADS, HIDDEN_DIM, N_LAYERS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(EPOCHS):
        random.shuffle(qa_pairs)
        pbar = tqdm(range(0, len(qa_pairs), BATCH_SIZE), desc=f"Эпоха {epoch + 1}/{EPOCHS}")
        for i in pbar:
            batch = qa_pairs[i:i + BATCH_SIZE]
            batch_ids = []
            max_l = 0
            for q, a in batch:
                ids = [word_to_idx.get(w, 1) for w in q.split() + ["<SOS>"] + a.split() + ["<EOS>"]]
                batch_ids.append(ids)
                max_l = max(max_l, len(ids))

            # ИСПРАВЛЕННАЯ СТРОКА:
            padded = [ids + [0] * (max_l - len(ids)) for ids in batch_ids]

            ids_tensor = torch.tensor(padded, dtype=torch.long).to(DEVICE)
            optimizer.zero_grad()
            output = model(ids_tensor[:, :-1])
            target = ids_tensor[:, 1:]
            loss = criterion(output.reshape(-1, VOCAB_SIZE), target.reshape(-1))
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    print("Обучение завершено. Сохраняю модель...")
    torch.save(model.state_dict(), MODEL_FILE)
    with open(VOCAB_FILE, "wb") as f:
        pickle.dump({"word_to_idx": word_to_idx, "idx_to_word": idx_to_word}, f)


# --- 4. Чат ---
def chat(text):
    model.eval()
    tokens = text.lower().split()
    ids = [word_to_idx.get(w, 1) for w in tokens] + [word_to_idx["<SOS>"]]
    for _ in range(20):
        curr_tensor = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = model(curr_tensor)[:, -1, :]

        # Температура 0.3 для точности
        probs = F.softmax(logits / 0.1, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        if next_token == word_to_idx["<EOS>"] or next_token == 0: break
        ids.append(next_token)

    try:
        sos_idx = ids.index(word_to_idx["<SOS>"])
        res_ids = ids[sos_idx + 1:]
        return " ".join([idx_to_word.get(i, "") for i in res_ids]).capitalize()
    except:
        return "..."


print("\n--- Чат запущен! ---")
while True:
    u_in = input("Вы: ")
    if u_in.lower() in ['exit', 'stop', 'выход']: break
    print(f"puppyGPT: {chat(u_in)}")

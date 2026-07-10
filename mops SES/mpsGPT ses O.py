import json, os, datetime

class SES_Neural_Final:
    def __init__(self, data_file="ses_dataset.json", model_file="ses_model.json"):
        self.data_file = data_file
        self.model_file = model_file
        self.learning_rate = 0.05
        self.storage = self._load_model()
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [(item['q'], item['a']) for item in data['training_data']]
        return [("ошибка", "Датасет не найден!")]

    def _load_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, "r") as f: return json.load(f)
        return {"weights": {}, "vocab": []}

    def tokenize(self, text):
        return [w.strip("?!.,") for w in text.lower().split() if len(w) > 2]

    def train(self, epochs=500):
        print(f"[*] Начало глубокого обучения (JSON Source)...")
        # Сбор словаря из JSON
        words = []
        for q, a in self.dataset: words.extend(self.tokenize(q))
        self.storage["vocab"] = list(set(words))
        
        for w in self.storage["vocab"]:
            if w not in self.storage["weights"]:
                self.storage["weights"][w] = [0.0] * len(self.dataset)

        for epoch in range(epochs):
            err = 0
            for i, (q, a) in enumerate(self.dataset):
                tks = self.tokenize(q)
                if not tks: continue
                pred = sum(self.storage["weights"][t][i] for t in tks) / len(tks)
                diff = 1.0 - pred
                err += abs(diff)
                for t in tks: self.storage["weights"][t][i] += diff * self.learning_rate
            if epoch % 100 == 0: print(f"Эпоха {epoch} | Ошибка: {err:.5f}")

        with open(self.model_file, "w") as f: json.dump(self.storage, f)
        print("[+] Модель сохранена. Готова к работе.")

    def chat(self):
        print("\n--- SES NEURAL INTERFACE v1.1 ---")
        while True:
            u = input("\n[Ввод]: ").lower()
            if u in ['exit', 'выход']: break
            tks = self.tokenize(u)
            if not tks: continue
            
            scores = [0.0] * len(self.dataset)
            for i in range(len(self.dataset)):
                valid = [t for t in tks if t in self.storage["weights"]]
                if valid:
                    scores[i] = sum(self.storage["weights"][t][i] for t in valid) / len(valid)
            
            idx = scores.index(max(scores))
            if scores[idx] > 0.4:
                print(f"[SES]: {self.dataset[idx][1]}")
            else:
                print("[SES]: Контекст не ясен. Нужно больше данных.")

# Запуск
ses = SES_Neural_Final()
ses.train(epochs=500)
ses.chat()

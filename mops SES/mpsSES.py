import json, os, datetime, random, math

class SES_Strict_Core:
    def __init__(self, data_file="ses_dataset.json", model_file="ses_model.json"):
        self.data_file = data_file
        self.model_file = model_file
        self.learning_rate = 0.02
        self.temperature = 0.2  # Понижена для исключения "галлюцинаций"
        
        self.dataset = self._load_data()
        self.storage = self._load_model()

    def _load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [(item['q'], item['a']) for item in data['training_data']]
        return [("привет", "Система SES готова.")]

    def _load_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, "r") as f: return json.load(f)
        return {"weights": {}, "vocab": []}

    def tokenize(self, text):
        return [w.strip("?!.,") for w in text.lower().split() if len(w) > 2]

    def softmax_sampling(self, scores):
        """Механизм температуры с защитой от переполнения"""
        # Т = 0.2 делает выбор почти детерминированным (строгим)
        max_s = max(scores)
        # Применяем масштабирование: чем меньше T, тем выше разрыв между вероятностями
        exp_scores = [math.exp((s - max_s) / self.temperature) for s in scores]
        sum_exp = sum(exp_scores)
        probs = [s / sum_exp for s in exp_scores]
        
        r, cum = random.random(), 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum: return i
        return scores.index(max_s)

    def train(self, epochs=500):
        print(f"[*] Закрепление векторов: {len(self.dataset)} пакетов...")
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
                pred = sum(self.storage["weights"][t][i] for t in tks if t in self.storage["weights"]) / len(tks)
                diff = 1.0 - pred
                err += abs(diff)
                for t in tks: self.storage["weights"][t][i] += diff * self.learning_rate
            if epoch % 250 == 0: print(f"Эпоха {epoch} | Стабильность: {1.0 - err:.4f}")
        
        with open(self.model_file, "w") as f: json.dump(self.storage, f)

    def chat(self):
        print(f"\n--- SES v1.3.1 (STRICT MODE | Temp: {self.temperature}) ---")
        print("Команды: /temp [число] - изменить креативность")
        
        while True:
            u = input("\n[Вы]: ").lower().strip()
            if u in ['exit', 'выход']: break
            
            # Смена температуры на лету
            if u.startswith("/temp"):
                try:
                    self.temperature = float(u.split()[1])
                    print(f"[SYSTEM]: Температура установлена на {self.temperature}")
                except: print("[!] Ошибка формата. Пример: /temp 0.1")
                continue

            tks = self.tokenize(u)
            if not tks: continue
            
            scores = [0.0] * len(self.dataset)
            for i in range(len(self.dataset)):
                matches = [t for t in tks if t in self.storage["weights"]]
                if matches:
                    scores[i] = sum(self.storage["weights"][t][i] for t in matches) / len(tks)

            if max(scores) > 0.15:
                idx = self.softmax_sampling(scores)
                print(f"[SES]: {self.dataset[idx][1]} (Conf: {max(scores):.2f})")
            else:
                print("[SES]: Контекст не ясен.")

# Запуск
ses = SES_Strict_Core()
ses.train(epochs=1000)
ses.chat()

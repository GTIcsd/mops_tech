import json, os, datetime, random, math

class SES_SmartSync:
    def __init__(self, data_file="ses_dataset.json", model_file="ses_model.json"):
        self.data_file = data_file
        self.model_file = model_file
        self.learning_rate = 0.03 # Чуть меньше, чтобы не "сломать" старые веса
        self.temperature = 0.6
        self.dataset = self._load_dataset()
        self.storage = self._load_and_sync()

    def _load_dataset(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [(item['q'], item['a']) for item in data['training_data']]
        return []

    def _load_and_sync(self):
        """Синхронизация старой модели с новым датасетом без удаления"""
        if not os.path.exists(self.model_file):
            return {"weights": {}, "vocab": []}
            
        with open(self.model_file, "r") as f:
            model = json.load(f)
            
        # ПРОВЕРКА СИНХРОНИЗАЦИИ
        expected_size = len(self.dataset)
        for word, weights in model["weights"].items():
            current_size = len(weights)
            if current_size < expected_size:
                # Добавляем новые ячейки памяти для новых вопросов
                model["weights"][word] += [0.0] * (expected_size - current_size)
                print(f"[*] Узел '{word}' расширен до {expected_size} параметров.")
            elif current_size > expected_size:
                # Обрезаем, если датасет уменьшился
                model["weights"][word] = weights[:expected_size]
        return model

    def tokenize(self, text):
        return [w.strip("?!.,") for w in text.lower().split() if len(w) > 2]

    def softmax_sampling(self, scores):
        # Ограничиваем значения для стабильности math.exp
        max_s = max(scores)
        exp_scores = [math.exp((s - max_s) / self.temperature) for s in scores]
        sum_exp = sum(exp_scores)
        probs = [s / sum_exp for s in exp_scores]
        
        r, cum = random.random(), 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum: return i
        return scores.index(max(scores))

    def train(self, epochs=500):
        print(f"[*] Дообучение модели на {len(self.dataset)} примерах...")
        words = []
        for q, a in self.dataset: words.extend(self.tokenize(q))
        self.storage["vocab"] = list(set(words))
        
        for w in self.storage["vocab"]:
            if w not in self.storage["weights"]:
                self.storage["weights"][w] = [0.0] * len(self.dataset)

        for epoch in range(epochs):
            total_err = 0
            for i, (q, a) in enumerate(self.dataset):
                tks = self.tokenize(q)
                if not tks: continue
                
                # Прямой проход
                active_weights = [self.storage["weights"][t][i] for t in tks if t in self.storage["weights"]]
                pred = sum(active_weights) / len(tks) if active_weights else 0
                
                diff = 1.0 - pred
                total_err += abs(diff)
                
                # Корректировка
                for t in tks:
                    self.storage["weights"][t][i] += diff * self.learning_rate
            
            if epoch % 100 == 0:
                print(f"Эпоха {epoch} | Ошибка: {total_err:.5f}")
        
        with open(self.model_file, "w") as f:
            json.dump(self.storage, f)
        print("[+] Интеллект сохранен и обновлен.")

    def chat(self):
        print(f"\n--- SES v1.2.2 HYBRID (Temp: {self.temperature}) ---")
        while True:
            u = input("\n[Вы]: ").lower().strip()
            if u in ['exit', 'выход']: break
            tks = self.tokenize(u)
            if not tks: continue
            
            scores = [0.0] * len(self.dataset)
            found_known_words = False
            
            for i in range(len(self.dataset)):
                matches = [t for t in tks if t in self.storage["weights"]]
                if matches:
                    found_known_words = True
                    # Считаем "силу" активации
                    scores[i] = sum(self.storage["weights"][t][i] for t in matches) / len(tks)

            if found_known_words and max(scores) > 0.05: # Снизили порог для гибкости
                idx = self.softmax_sampling(scores)
                print(f"[SES]: {self.dataset[idx][1]} (Уверенность: {max(scores):.2f})")
            else:
                print("[SES]: Это новое для меня. Попробуй использовать другие слова или дообучи меня.")

# --- ЗАПУСК ---
ses = SES_SmartSync()
ses.train(epochs=500) # Она просто обновит веса, не стирая старые
ses.chat()

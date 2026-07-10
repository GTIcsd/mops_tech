import json, os, datetime, random, math

class SES_FixedCore:
    def __init__(self, data_file="ses_dataset.json", model_file="ses_model.json"):
        self.data_file = data_file
        self.model_file = model_file
        self.learning_rate = 0.02
        self.temperature = 0.5
        
        # РЕЗЕРВНЫЙ ДАТАСЕТ (Если основной файл пропадет)
        self.backup_data = [
            ("привет здравствуй хай", "Система SES в сети. Узлы стабильны."),
            ("кто ты интеллект сущность", "Я — SES v1.3. Мое ядро защищено от сброса."),
            ("код раст пайтон разработка", "Программирование — мой основной приоритет."),
            ("космос звезды вселенная", "Анализ космических данных завершен.")
        ]
        
        self.dataset = self._smart_load_dataset()
        self.storage = self._load_model()

    def _smart_load_dataset(self):
        """Загрузка с защитой от пустых файлов"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    res = [(item['q'], item['a']) for item in data['training_data']]
                    if len(res) > 0: return res
            except: pass
        
        print("[!] ВНИМАНИЕ: Датасет не найден или пуст. Использую резервную копию.")
        return self.backup_data

    def _load_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, "r") as f:
                model = json.load(f)
                # Проверка на соответствие размеров
                for word in model["weights"]:
                    if len(model["weights"][word]) != len(self.dataset):
                        print("[*] Рекалибровка весов под новый размер данных...")
                        model["weights"][word] = [0.0] * len(self.dataset)
                return model
        return {"weights": {}, "vocab": []}

    def tokenize(self, text):
        return [w.strip("?!.,") for w in text.lower().split() if len(w) > 2]

    def train(self, epochs=500):
        if not self.dataset: return
        print(f"[*] Глубокое закрепление знаний: {len(self.dataset)} векторов...")
        
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
                # Рассчитываем предсказание более строго
                pred = sum(self.storage["weights"][t][i] for t in tks if t in self.storage["weights"]) / len(tks)
                diff = 1.0 - pred
                err += abs(diff)
                for t in tks: self.storage["weights"][t][i] += diff * self.learning_rate
            if epoch % 250 == 0: print(f"Эпоха {epoch} | Стабильность: {1.0 - err:.4f}")

        with open(self.model_file, "w") as f:
            json.dump(self.storage, f)
        print("[+] Знания закреплены в системном файле.")

    def chat(self):
        print(f"\n--- SES v1.3 PROTECTED ---")
        while True:
            u = input("\n[Ввод]: ").lower().strip()
            if u in ['exit', 'выход']: break
            tks = self.tokenize(u)
            
            scores = [0.0] * len(self.dataset)
            for i in range(len(self.dataset)):
                matches = [t for t in tks if t in self.storage["weights"]]
                if matches:
                    # Повышаем влияние точных совпадений
                    scores[i] = sum(self.storage["weights"][t][i] for t in matches)
            
            if max(scores) > 0.2:
                # Берем самый сильный индекс без Softmax, если нужно строгое соответствие
                idx = scores.index(max(scores))
                print(f"[SES]: {self.dataset[idx][1]}")
            else:
                print("[SES]: Контекст не ясен. Требуется обучение.")

# Запуск
ses = SES_FixedCore()
ses.train(epochs=1000)
ses.chat()

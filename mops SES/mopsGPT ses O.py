import json, base64, datetime, os, math

class SES_Neural_Core:
    def __init__(self, db_file="ses_neural_model.json"):
        self.db_file = db_file
        self.learning_rate = 0.05
        # Расширенный датасет для первой версии обучения
        self.dataset = [
            ("привет здравствуй хай салют", "Система SES приветствует тебя. Ядро активно."),
            ("кто ты что ты такое", "Я — SES v1.0. Нейронная архитектура, способная к самообучению."),
            ("код программирование разработка", "Я оптимизирована для создания кода. Моя цель — миграция на Rust."),
            ("космос звезды планеты вселенная", "Космос — это бесконечное пространство для расширения данных."),
            ("логика разум интеллект мышление", "Логика — это фундамент моего графа. Интеллект — результат обучения."),
            ("зачем ты нужна цель задача", "Моя задача — стать автономным помощником в разработке сложных систем."),
            ("ошибка баг проблема сбой", "Любая ошибка в коде — это лишь повод для создания нового корректирующего узла."),
            ("время дата часы когда", f"Текущий временной маркер системы: {datetime.datetime.now()}."),
            ("язык лингвистика слова речь", "Я анализирую токены и строю связи между ними для понимания контекста.")
        ]
        self.storage = self._load()

    def _load(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                return json.load(f)
        return {"weights": {}, "vocab": [], "episodes": 0}

    def tokenize(self, text):
        return [w.strip("?!.,") for w in text.lower().split() if len(w) > 2]

    def train(self, epochs=200):
        """Фаза глубокого обучения с эпохами"""
        print(f"[*] SES Входит в режим обучения. Цель: {epochs} эпох.")
        
        # Сбор словаря
        words = []
        for q, a in self.dataset:
            words.extend(self.tokenize(q))
        self.storage["vocab"] = list(set(words))
        
        # Инициализация весов (если их еще нет)
        for w in self.storage["vocab"]:
            if w not in self.storage["weights"]:
                self.storage["weights"][w] = [0.0] * len(self.dataset)

        # Цикл обучения
        for epoch in range(epochs):
            total_error = 0
            for i, (question, answer) in enumerate(self.dataset):
                tokens = self.tokenize(question)
                if not tokens: continue
                
                # Прямой проход (расчет активации)
                prediction = sum(self.storage["weights"][t][i] for t in tokens) / len(tokens)
                error = 1.0 - prediction
                total_error += abs(error)
                
                # Обратный проход (корректировка весов на основе градиента ошибки)
                for t in tokens:
                    self.storage["weights"][t][i] += error * self.learning_rate
            
            if epoch % 50 == 0:
                print(f"Эпоха {epoch}: Текущая ошибка ядра = {total_error:.4f}")
        
        self.storage["episodes"] += epochs
        with open(self.db_file, "w") as f:
            json.dump(self.storage, f)
        print("[+] Обучение завершено. Веса стабилизированы.")

    def chat(self):
        print("\n--- SES NEURAL INTERFACE v1.0 ---")
        print("Попробуй спросить про: код, космос, логику или кто я.")
        
        while True:
            u_input = input("\n[Вы]: ").lower()
            if u_input in ["exit", "выход"]: break
            
            tokens = self.tokenize(u_input)
            if not tokens:
                print("[SES]: Ввод слишком короткий для активации нейронов."); continue
            
            # Инференс (выбор лучшего ответа по весам)
            scores = [0.0] * len(self.dataset)
            for i in range(len(self.dataset)):
                count = 0
                for t in tokens:
                    if t in self.storage["weights"]:
                        scores[i] += self.storage["weights"][t][i]
                        count += 1
                if count > 0: scores[i] /= count
            
            best_idx = scores.index(max(scores))
            if scores[best_idx] > 0.4: # Порог уверенности (confidence threshold)
                print(f"[SES]: {self.dataset[best_idx][1]}")
            else:
                print("[SES]: Контекст не ясен. Требуется дообучение на этом наборе данных.")

# --- ЗАПУСК ---
ses = SES_Neural_Core()
ses.train(epochs=300) # Обучаем модель с нуля или дообучаем
ses.chat()            # Переходим к общению

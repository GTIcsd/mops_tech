import json, base64, datetime, os

class SES_DeepLearner:
    def __init__(self, db_file="ses_snapshot.json"):
        self.db_file = db_file
        self.storage = self._load()
        print(f"[*] SES v0.8 Online. База знаний: {len(self.storage.get('graph', {}))} узлов.")

    def _load(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r") as f:
                    d = json.load(f)
                    for k in ["graph", "memory"]:
                        if k not in d: d[k] = {} if k == "graph" else []
                    return d
            except: pass
        return {"graph": {}, "memory": []}

    def _save(self):
        with open(self.db_file, "w") as f:
            json.dump(self.storage, f, indent=4)

    def validator(self, code):
        forbidden = ["os.", "sys.", "subprocess", "eval(", "open("]
        return not any(f in code for f in forbidden)

    def architect(self, node_id, topic):
        """Создает узел-компетенцию"""
        code = (
            f"def execute(data, kernel):\n"
            f"    return f'[АНАЛИЗ_{node_id.upper()}]: Обработан контекст \"{{data}}\" по теме {topic}'"
        )
        if self.validator(code):
            self.storage["graph"][node_id] = base64.b64encode(code.encode()).decode()
            self._save()
            return True
        return False

    def train(self, data_list):
        """Массовое обучение на входящем потоке данных"""
        print(f"[*] Запуск процесса глубокого обучения ({len(data_list)} пакетов)...")
        
        # Расширенная карта распознавания тем
        patterns = {
            "phys": "Физика и энергия",
            "quant": "Квантовые вычисления",
            "lingua": "Лингвистика и языки",
            "neuro": "Нейросети и мозг",
            "astro": "Астрофизика",
            "chem": "Химия элементов",
            "arch": "Архитектура систем"
        }
        
        for text in data_list:
            for key, desc in patterns.items():
                if key in text.lower():
                    node_name = f"node_{key}"
                    if node_name not in self.storage["graph"]:
                        self.architect(node_name, desc)
            
            # Логируем опыт
            self.storage["memory"].append({
                "ts": str(datetime.datetime.now()),
                "data": text[:50]
            })
        self._save()
        print("[+] Обучение завершено успешно.")

    def chat(self):
        """Режим живого общения в консоли"""
        print("\n--- SES INTERFACE v0.8 ---")
        print("Введите запрос (или 'exit' для выхода, 'status' для статистики)")
        
        while True:
            user_input = input("\n[Ввод]: ")
            if user_input.lower() in ['exit', 'quit', 'выход']: break
            
            if user_input.lower() == 'status':
                print(f"[SES]: Узлов: {len(self.storage['graph'])}, Память: {len(self.storage['memory'])} записей.")
                continue

            # Поиск подходящих узлов для ответа
            responses = []
            for node_id in self.storage["graph"]:
                core_word = node_id.replace("node_", "")
                if core_word in user_input.lower():
                    raw_code = base64.b64decode(self.storage["graph"][node_id]).decode()
                    scope = {}
                    exec(raw_code, {}, scope)
                    responses.append(scope['execute'](user_input, self))
            
            if responses:
                print("\n".join(responses))
            else:
                print("[SES]: Прямых ассоциаций не найдено. Данные сохранены для фоновой эволюции.")
                self.storage["memory"].append({"ts": str(datetime.datetime.now()), "unhandled": user_input})
                self._save()

# --- ЗАПУСК И ОБУЧЕНИЕ ---

ses = SES_DeepLearner()

# Большой датасет для "прокачки"
big_data = [
    "Quantum computing uses qubits for parallel phys calculations",
    "Lingua research shows how neuro-pathways form during speech",
    "Astro-navigation requires complex arch structures in software",
    "Chemical reactions in neuro-transmitters affect mood",
    "The phys-ics of black holes is a key part of astro-logy",
    "Advanced arch for quant-um systems is being developed"
]

ses.train(big_data)

# Переход в режим общения
ses.chat()

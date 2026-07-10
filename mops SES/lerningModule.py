import json, base64, datetime, os

class SES_Universal:
    def __init__(self, db_file="ses_snapshot.json"):
        self.db_file = db_file
        self.storage = self._load()
        print(f"[*] SES v0.7.1 Online. Nodes: {len(self.storage.get('graph', {}))}")

    def _load(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r") as f:
                    d = json.load(f)
                    for k in ["graph", "memory", "metadata"]:
                        if k not in d: d[k] = {} if k != "memory" else []
                    return d
            except: pass
        return {"graph": {}, "memory": [], "metadata": {"status": "active"}}

    def _save(self):
        with open(self.db_file, "w") as f:
            json.dump(self.storage, f, indent=4)

    def validator(self, code):
        """Проверка кода перед интеграцией в ядро"""
        forbidden = ["os.", "sys.", "subprocess", "eval(", "open(", "__"]
        return not any(f in code for f in forbidden)

    def architect(self, node_id, topic):
        """Создает новый функциональный узел для конкретной темы"""
        code = (
            f"def execute(data, kernel):\n"
            f"    # Автоматический узел для темы: {topic}\n"
            f"    return f'ANALYSIS_{node_id.upper()} >> Входящий поток: {{data}}'"
        )
        if self.validator(code):
            self.storage["graph"][node_id] = base64.b64encode(code.encode()).decode()
            self._save()
            print(f"[+] Сформирован новый навык: {node_id}")
            return True
        return False

    def learn(self, data_stream):
        """Обучение на любых данных: ищет темы и создает узлы"""
        print(f"[*] Обработка данных: '{data_stream[:40]}...'")
        
        # Словарь потенциальных навыков (можно расширять бесконечно)
        knowledge_map = {
            "math": "Математические вычисления",
            "code": "Программирование и алгоритмы",
            "cook": "Кулинария и рецепты",
            "space": "Астрономия и космос",
            "music": "Музыкальная теория",
            "history": "Исторические факты",
            "bio": "Биология и медицина"
        }
        
        found_any = False
        for key, description in knowledge_map.items():
            if key in data_stream.lower():
                node_name = f"node_{key}"
                if node_name not in self.storage["graph"]:
                    self.architect(node_name, description)
                    found_any = True
        
        # Сохраняем сырой опыт в память
        self.storage["memory"].append({
            "ts": str(datetime.datetime.now()),
            "raw": data_stream,
            "learned": found_any
        })
        self._save()

    def chat(self, user_input):
        """Интерфейс прямого взаимодействия"""
        print(f"\n[USER]: {user_input}")
        cmd = user_input.lower()

        if "статус" in cmd:
            return f"[SES]: Граф: {len(self.storage['graph'])} узлов. Память: {len(self.storage['memory'])} событий."
        
        if "забудь" in cmd:
            self.storage = {"graph": {}, "memory": [], "metadata": {"status": "reset"}}
            self._save()
            return "[SES]: Память очищена. Перезагрузка графа."

        # Попытка вызвать узел по ключевому слову
        for node_id in self.storage["graph"]:
            topic = node_id.replace("node_", "")
            if topic in cmd:
                raw_code = base64.b64decode(self.storage["graph"][node_id]).decode()
                scope = {}
                exec(raw_code, {}, scope)
                return f"[SES]: Активирован модуль {topic}. " + scope['execute'](user_input, self)

        return "[SES]: Данные приняты. Я интегрирую эту информацию в latent-слой."

# --- ПРИМЕР РАЗНООБРАЗНОГО ОБУЧЕНИЯ ---

ses = SES_Universal()

# 1. Загружаем разные данные
diverse_data = [
    "Math is essential for space travel and orbit calculation",
    "Bio-engineering is the future of medicine",
    "The history of music starts with ancient bone flutes",
    "To cook a perfect steak, use high heat",
    "Python code is easy to read"
]

for item in diverse_data:
    ses.learn(item)

# 2. Общаемся
print(ses.chat("Какой сейчас статус?"))
print(ses.chat("Расскажи про math и space"))
print(ses.chat("Что ты думаешь про cook?"))

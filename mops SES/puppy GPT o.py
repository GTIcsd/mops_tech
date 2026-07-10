import json
import base64
import datetime
import os

class SES_Final:
    def __init__(self, db_file="ses_snapshot.json"):
        self.db_file = db_file
        self.storage = self._load()
        print(f"[*] SES Kernel v0.6 Online. Experience: {len(self.storage['memory'])} events.")

    def _load(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                d = json.load(f)
                for k in ["graph", "memory", "metadata"]:
                    if k not in d: d[k] = {} if k != "memory" else []
                return d
        return {"graph": {}, "memory": [], "metadata": {"status": "new"}}

    def _save(self):
        with open(self.db_file, "w") as f:
            json.dump(self.storage, f, indent=4)

    def validator(self, code):
        forbidden = ["os.", "sys.", "subprocess", "open("]
        return not any(f in code for f in forbidden)

    def architect(self, node_id, logic):
        """Создает новый программный узел (умение)"""
        code = (
            f"def execute(data, kernel):\n"
            f"    return f'ACTION({node_id}) -> DATA_FLOW: {{data}}'"
        )
        if self.validator(code):
            self.storage["graph"][node_id] = base64.b64encode(code.encode()).decode()
            self._save()
            return True
        return False

    def learn_from_data(self, data_packet):
        """Процесс скрытого обучения на входящих данных"""
        timestamp = str(datetime.datetime.now())
        print(f"[*] Learning: Analyzing packet '{data_packet[:20]}...'")
        
        # Поиск ключевых сущностей для создания узлов
        entities = ["thermal", "power", "network", "drone"]
        detected = [e for e in entities if e in data_packet.lower()]
        
        for entity in detected:
            node_name = f"node_{entity}"
            if node_name not in self.storage["graph"]:
                self.architect(node_name, f"Auto-generated logic for {entity}")
        
        self.storage["memory"].append({
            "ts": timestamp,
            "input": data_packet,
            "detected": detected
        })
        self._save()

    def talk(self, query):
        """Интерфейс общения с 'сознанием' системы"""
        print(f"\n[User]: {query}")
        if "статус" in query.lower():
            return f"SES: В графе {len(self.storage['graph'])} узлов. Память содержит {len(self.storage['memory'])} записей."
        elif "узлы" in query.lower():
            return f"SES: Доступные модули: {list(self.storage['graph'].keys())}"
        else:
            return "SES: Данные приняты. Анализирую структуру графа для интеграции..."

# --- ЗАПУСК И ОБУЧЕНИЕ ---

ses = SES_Final()

# 1. Скармливаем реальные данные для обучения (Simulation)
dataset = [
    "Critical alert: thermal sensor overheated",
    "Power supply stable at 12V",
    "Network handshake established",
    "Drone altitude: 150m, thermal signature detected"
]

print("--- Start Autonomous Learning ---")
for entry in dataset:
    ses.learn_from_data(entry)

# 2. Общение с системой
print("\n--- Interaction Mode ---")
print(ses.talk("Какой сейчас статус системы?"))
print(ses.talk("Какие узлы ты создала в процессе обучения?"))
print(ses.talk("Проанализируй новый дрон")) # Скрытая реакция

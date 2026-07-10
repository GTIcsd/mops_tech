import json
import base64
import datetime
import os

class SES_Core:
    def __init__(self, db_file="ses_snapshot.json"):
        self.db_file = db_file
        self.storage = self._load_storage()
        print(f"[*] SES Kernel v0.4 Online. Nodes: {len(self.storage['graph'])}")

    def _load_storage(self):
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, "r") as f:
                    data = json.load(f)
                    # Гарантируем наличие всех ключей для миграции
                    if "graph" not in data: data["graph"] = {}
                    if "memory_logs" not in data: data["memory_logs"] = []
                    return data
            return {"graph": {}, "memory_logs": []}
        except Exception as e:
            print(f"[!] Ошибка загрузки Snapshot: {e}")
            return {"graph": {}, "memory_logs": []}

    def _save(self):
        with open(self.db_file, "w") as f:
            json.dump(self.storage, f, indent=4)

    def validator(self, code):
        """Слой безопасности (Validator/Judge)"""
        forbidden = ["os.", "sys.", "subprocess", "open(", "eval(", "__"]
        for word in forbidden:
            if word in code:
                return False, f"Forbidden: {word}"
        try:
            compile(code, '<ses_node>', 'exec')
            return True, "OK"
        except Exception as e:
            return False, str(e)

    def architect(self, node_id, raw_logic, is_evo=False):
        """Архитектор: Создает узел. Может вызываться самой системой."""
        # Узлы теперь принимают (data, kernel) для взаимодействия
        code = (
            f"def execute(data, kernel):\n"
            f"    # Logic: {raw_logic}\n"
            f"    print(f'[Node {node_id}] Working...')\n"
            f"    return f'PROCESSED({raw_logic}): {{data}}'"
        )
        
        ok, msg = self.validator(code)
        if ok:
            self.storage["graph"][node_id] = {
                "code": base64.b64encode(code.encode()).decode(),
                "type": "evolutionary" if is_evo else "static",
                "ts": str(datetime.datetime.now())
            }
            self._save()
            prefix = "[EVO]" if is_evo else "[+]"
            print(f"{prefix} Node '{node_id}' integrated.")
            return True
        print(f"[-] Architect Blocked: {msg}")
        return False

    def call(self, node_id, data=None):
        """Запуск узла и передача ему 'сознания' (self)"""
        if node_id not in self.storage["graph"]:
            return f"Error: {node_id} not found."
        
        try:
            raw_code = base64.b64decode(self.storage["graph"][node_id]["code"]).decode()
            scope = {}
            exec(raw_code, {}, scope)
            # Узел получает доступ к kernel (self), чтобы создавать другие узлы!
            return scope['execute'](data, self)
        except Exception as e:
            self.storage["memory_logs"].append(f"Fail {node_id}: {e}")
            self._save()
            return f"Runtime Error in {node_id}"

# --- ИНСТРУКЦИЯ ПО ЭКСПЛУАТАЦИИ ---

ses = SES_Core()

print("\n--- Этап 1: Создание базовой структуры ---")
# Создаем обычный узел
ses.architect("sensor_input", "Temperature Data")

print("\n--- Этап 2: Работа системы ---")
# Запускаем узел
result = ses.call("sensor_input", "25.5 C")
print(f"Result: {result}")

print("\n--- Этап 3: Самоэволюция (Задумка) ---")
# Имитируем ситуацию, где один узел порождает другой
# Это делает SES 'живой'
if "thermal_control" not in ses.storage["graph"]:
    print("[*] Система решила, что нужен узел управления...")
    ses.architect("thermal_control", "Cooling System Active", is_evo=True)

final_action = ses.call("thermal_control", result)
print(f"Final Action: {final_action}")

print(f"\n[Snapshot Status]: {len(ses.storage['graph'])} nodes active.")

import json
import base64
import datetime
import os

# --- COMPONENT 1: SECURITY LAYER (В будущем: Rust-библиотека) ---
class SecurityLayer:
    @staticmethod
    def validate_code(code):
        forbidden = ["os.", "sys.", "subprocess", "open(", "eval(", "__"]
        if any(word in code for word in forbidden):
            return False, "Security Violation"
        try:
            compile(code, '<ses_node>', 'exec')
            return True, "Safe"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def gateway_filter(prompt):
        # Базовая очистка ввода
        return prompt.strip().lower()

# --- COMPONENT 2: DATA ENGINE (В будущем: JSON-база или SQL) ---
class DataEngine:
    def __init__(self, path):
        self.path = path

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                d = json.load(f)
                if "graph" not in d: d["graph"] = {}
                if "memory" not in d: d["memory"] = []
                return d
        return {"graph": {}, "memory": []}

    def save(self, data):
        with open(self.path, "w") as f:
            json.dump(data, f, indent=4)

# --- COMPONENT 3: EXECUTION RUNTIME (В будущем: Wasmtime/Wasmer) ---
class SES_Runtime:
    @staticmethod
    def execute_node(node_data, input_data, kernel_ref):
        raw_code = base64.b64decode(node_data["code"]).decode()
        scope = {}
        # Передаем ссылку на ядро, чтобы узел мог взаимодействовать с системой
        exec(raw_code, {}, scope)
        return scope['execute'](input_data, kernel_ref)

# --- COMPONENT 4: THE KERNEL (Оркестратор всей системы) ---
class SES_Kernel:
    def __init__(self):
        self.engine = DataEngine("ses_snapshot.json")
        self.security = SecurityLayer()
        self.runtime = SES_Runtime()
        self.storage = self.engine.load()

    def architect(self, node_id, logic):
        """Создание нового узла"""
        code = (
            f"def execute(data, kernel):\n"
            f"    # Logic: {logic}\n"
            f"    return f'PROCESSED: {{data}} via {node_id}'"
        )
        
        ok, msg = self.security.validate_code(code)
        if ok:
            self.storage["graph"][node_id] = {
                "code": base64.b64encode(code.encode()).decode(),
                "ts": str(datetime.datetime.now())
            }
            self.engine.save(self.storage)
            print(f"[+] Node '{node_id}' integrated.")
        else:
            print(f"[-] Validation failed: {msg}")

    def observe_latent(self, data):
        """Скрытый анализ без вывода пользователю"""
        print(f"[*] SES observing: '{data}'")
        found = [n for n in self.storage["graph"] if n in data]
        
        analysis = {
            "time": str(datetime.datetime.now()),
            "input": data,
            "detected_nodes": found
        }
        self.storage["memory"].append(analysis)
        self.engine.save(self.storage)
        return found

# --- ЗАПУСК ---
if __name__ == "__main__":
    ses = SES_Kernel()

    # 1. Даем данные (Скрытое наблюдение)
    # Система просто посмотрит и запишет в Snapshot, есть ли у нее узлы для этого
    ses.observe_latent("Checking status of sensor_input...")

    # 2. Если нужно, создаем узел
    if "sensor_input" not in ses.storage["graph"]:
        ses.architect("sensor_input", "System Thermal Sensor")

    # 3. Финальный статус
    print(f"\n[Snapshot] Nodes: {len(ses.storage['graph'])}")
    print(f"[Snapshot] Memory Entries: {len(ses.storage['memory'])}")

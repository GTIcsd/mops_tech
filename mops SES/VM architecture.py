import json
import base64
import datetime
import traceback

class SES_Full_Prototype:
    def __init__(self, db_file="ses_snapshot.json"):
        self.db_file = db_file
        self.storage = self._load_storage()
        print(f"[*] SES Kernel Loaded. Nodes: {len(self.storage['graph'])}")

    def _load_storage(self):
        try:
            with open(self.db_file, "r") as f:
                return json.load(f)
        except:
            return {
                "metadata": {"ver": "0.3-alpha", "engine": "Python-Pydroid"},
                "graph": {},
                "memory_logs": []
            }

    def _save(self):
        with open(self.db_file, "w") as f:
            json.dump(self.storage, f, indent=4)

    def gateway_filter(self, prompt):
        """Мини-фильтр: Анализирует ввод на наличие шума или угроз"""
        toxic_keywords = ["delete system", "format c", "drop table"]
        if any(k in prompt.lower() for k in toxic_keywords):
            return False, "GATEWAY_REJECT: Unsafe prompt pattern."
        return True, prompt.strip()

    def validator(self, code):
        """Оценщик (Validator/Judge): Проверка на безопасность и синтаксис"""
        # Эмуляция строгой Rust-проверки
        forbidden = ["os.", "sys.", "subprocess", "eval(", "getattr", "__", "open("]
        for word in forbidden:
            if word in code:
                return False, f"VALIDATOR_REJECT: Forbidden keyword '{word}'"
        try:
            compile(code, '<ses_node>', 'exec')
            return True, "Code is safe"
        except Exception as e:
            return False, f"VALIDATOR_REJECT: Syntax Error -> {e}"

    def architect(self, node_id, raw_logic, deps=[]):
        """Архитектор: Проектирует новый программный узел"""
        # Формируем структуру функции
        # В реальной SES здесь LLM пишет сложный алгоритм
        code = (
            f"def execute(context):\n"
            f"    # Logic: {raw_logic}\n"
            f"    # Dependencies: {deps}\n"
            f"    res = '{raw_logic}'.upper()\n"
            f"    return res"
        )
        
        is_ok, msg = self.validator(code)
        if is_ok:
            encoded = base64.b64encode(code.encode()).decode()
            self.storage["graph"][node_id] = {
                "code": encoded,
                "deps": deps,
                "created": str(datetime.datetime.now())
            }
            self._save()
            print(f"[+] Node '{node_id}' integrated successfully.")
            return True
        else:
            print(f"[-] Architect failure: {msg}")
            return False

    def run_with_evolution(self, node_id):
        """Запуск узла с механизмом самоисправления (Evolver)"""
        if node_id not in self.storage["graph"]:
            return "Error: Node not found in Snapshot."

        try:
            # Декодируем и исполняем
            raw_code = base64.b64decode(self.storage["graph"][node_id]["code"]).decode()
            scope = {}
            exec(raw_code, {}, scope)
            return scope['execute'](self.storage)
            
        except Exception as e:
            # Слой Evolver: если узел упал, создаем 'исправленную' версию
            error_info = f"Crash in {node_id}: {str(e)}"
            print(f"[!] {error_info}")
            
            new_node_id = f"{node_id}_evo_{len(self.storage['graph'])}"
            print(f"[*] Evolving... Creating stable node: {new_node_id}")
            
            # Архитектор создает "заплатку"
            self.architect(new_node_id, f"Recovered version of {node_id} (Fixed Error)")
            
            # Запись в серийную преемственность
            self.storage["memory_logs"].append({
                "failed_node": node_id,
                "error": str(e),
                "timestamp": str(datetime.datetime.now())
            })
            self._save()
            
            return f"Evolution Triggered. Current task rerouted to {new_node_id}."

# --- ПРИМЕР РАБОТЫ ---
ses = SES_Full_Prototype()

# 1. Проход через Gateway и создание рабочего узла
status, prompt = ses.gateway_filter("Create data_sync module")
if status:
    ses.architect("sync_node", prompt)

# 2. Попытка создать вредоносный узел (Валидатор заблокирует)
ses.architect("malware", "import os; os.system('echo hi')")

# 3. Эмуляция ошибки и самоэволюции
# Специально создаем узел, который выдаст ошибку Runtime (например, TypeError)
# В коде выше execute(context) принимает 1 аргумент, мы вызовем его корректно,
# но представим, что логика внутри была бы сложнее.
print(f"\nExecution result: {ses.run_with_evolution('sync_node')}")

# 4. Проверка Snapshot
print(f"\nSnapshot Status: {len(ses.storage['graph'])} nodes, {len(ses.storage['memory_logs'])} incidents.")

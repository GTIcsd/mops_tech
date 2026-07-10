import json
import base64

class SES_Kernel:
    def __init__(self, snapshot_path="brain.json"):
        self.snapshot_path = snapshot_path
        # Динамический граф навыков (название: код)
        self.skills = self.load_snapshot()

    def load_snapshot(self):
        try:
            with open(self.snapshot_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"version": 1.0, "nodes": {}}

    def save_snapshot(self):
        with open(self.snapshot_path, "w") as f:
            json.dump(self.skills, f, indent=2)

    def validator(self, code):
        """Эмуляция Rust-валидатора: проверка на 'запрещенку' и синтаксис."""
        forbidden = ["import os", "open(", "eval", "sys"]
        if any(f in code for f in forbidden):
            return False, "Security Breach: Forbidden calls detected."
        try:
            compile(code, "<string>", "exec")
            return True, "Success"
        except Exception as e:
            return False, f"Syntax/Logic Error: {e}"

    def architect(self, task_name, logic_prompt):
        """Создает микро-модуль (скрипт) под задачу."""
        # В реальности здесь запрос к LLM. Здесь — генерация шаблона.
        generated_code = f"def execute():\n    return '{logic_prompt.upper()}'"
        
        is_valid, msg = self.validator(generated_code)
        if is_valid:
            self.skills["nodes"][task_name] = base64.b64encode(generated_code.encode()).decode()
            print(f"[*] Node '{task_name}' integrated into graph.")
            self.save_snapshot()
        else:
            print(f"[!] Validation failed: {msg}")

    def run_skill(self, task_name):
        """Запуск модуля из графа."""
        if task_name in self.skills["nodes"]:
            code = base64.b64decode(self.skills["nodes"][task_name]).decode()
            local_vars = {}
            exec(code, {}, local_vars)
            return local_vars['execute']()
        return "Skill not found."

# --- Тест-драйв ---
ses = SES_Kernel()

# 1. Архитектор создает навык
ses.architect("data_processor", "Processed successfull data flow")

# 2. Система 'вспоминает' навык и исполняет его
print(f"Output: {ses.run_skill('data_processor')}")

# 3. Попытка внедрить вредоносный код (отсекается Валидатором)
ses.architect("hacker_tool", "import os; os.system('rm -rf /')")

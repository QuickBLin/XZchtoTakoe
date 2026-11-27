"""Стенд воспроизведения кадров. F8 переключает сценарий в живом Tk."""
import ast
import sys
import tempfile
from pathlib import Path
from tkinter import Frame, Label, Text, Tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import ALL_QUEUES, Desk
from journal import Journal
from triage import QUEUES, fit_model, load_policy, read_split

COLOR = "#102b34"


def excerpt(filename, symbol, length=19):
    source = (ROOT / filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(item for item in ast.walk(tree)
                if getattr(item, "name", None) == symbol and hasattr(item, "lineno"))
    lines = source.splitlines()
    end = min(node.end_lineno, node.lineno + length - 1)
    code = "\n".join(f"{i:>3}   {lines[i - 1]}" for i in range(node.lineno, end + 1))
    return f"{filename}   /   {symbol}   /   строки {node.lineno}–{end}", code


class Frames:
    def __init__(self, window, db):
        self.window = window
        self.current = 0
        self.desk = Desk(window, fit_model(read_split("train")), load_policy(), Journal(db))
        window.geometry("1180x845+65+55")
        window.attributes("-topmost", True)
        window.bind_all("<F8>", self.next)
        self.scenes = [
            ("01-automatic", lambda: self.submit("Доступ к порталу")),
            ("02-review", lambda: self.submit("Неясное обращение")),
            ("03-correction", self.correct),
            ("04-import", lambda: self.desk.import_csv(ROOT / "examples" / "demo_batch.csv")),
            ("05-duplicate", lambda: self.desk.import_csv(ROOT / "examples" / "demo_batch.csv")),
            ("06-rejected", lambda: self.desk.import_csv(ROOT / "examples" / "rejected_batch.csv")),
            ("07-error", lambda: self.submit("Две проблемы")),
            ("08-filter", self.filter_review),
            ("09-tests", lambda: self.log_panel("Системные проверки · фактический вывод", "run_system.log")),
            ("10-experiment", lambda: self.log_panel("Сравнение методов · фактический вывод", "run_experiments.log")),
            ("11-code-model", lambda: self.code_panel("triage.py", "fit_model")),
            ("12-code-route", lambda: self.code_panel("triage.py", "route")),
            ("13-code-batch", lambda: self.code_panel("journal.py", "add_batch", 22)),
            ("14-code-test", lambda: self.code_panel("test_extended.py", "test_sql_failure_rolls_back_entire_batch")),
        ]
        self.show()

    def submit(self, name):
        self.desk.example.set(name)
        self.desk.use_example(None)
        self.desk.submit_button.invoke()

    def correct(self):
        self.desk.table.selection_set("2")
        self.desk.select_row()
        self.desk.queue_choice.set(QUEUES["access"])
        self.desk.move_button.invoke()

    def filter_review(self):
        self.desk.queue_filter.set(QUEUES["review"])
        self.desk.refresh()

    def log_panel(self, title, name):
        self.viewer(title, (ROOT / "results" / name).read_text(encoding="utf-8"), "Вывод сохранён из завершённого прогона")

    def code_panel(self, filename, symbol, length=19):
        heading, code = excerpt(filename, symbol, length)
        self.viewer(heading, code, "Фрагмент исходного файла текущей версии")

    def viewer(self, heading, content, subtitle):
        for item in self.window.winfo_children():
            item.destroy()
        frame = Frame(self.window, bg=COLOR, padx=29, pady=23)
        frame.pack(expand=True, fill="both")
        Label(frame, text=heading, bg=COLOR, fg="#f3fbfb",
              font=("Segoe UI", 21, "bold"), anchor="w").pack(fill="x")
        Label(frame, text=subtitle, bg=COLOR, fg="#8fe4cc",
              font=("Segoe UI", 12), anchor="w").pack(fill="x", pady=(5, 18))
        text = Text(frame, wrap="none", bg="#162127", fg="#d9f1ec",
                    font=("Consolas", 11), padx=18, pady=14, relief="flat")
        text.pack(fill="both", expand=True)
        text.insert("1.0", content)
        text.configure(state="disabled")
        text.focus_set()

    def show(self):
        name, action = self.scenes[self.current]
        action()
        self.window.title(f"ИИС 2.0 · Стенд · {name}")
        self.window.update_idletasks()

    def next(self, _event):
        if self.current < len(self.scenes) - 1:
            self.current += 1
            self.show()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        root = Tk()
        Frames(root, Path(folder) / "evidence.sqlite3")
        root.mainloop()

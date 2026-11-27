"""Окно оператора. Запуск командой python app.py из папки проекта."""
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from batch import describe, import_file
from journal import Journal
from assistant import CaseIndex, operator_brief
from triage import CATEGORIES, QUEUES, REVIEW, baseline, fit_model, load_policy
from triage import predict, read_split, route, validate_text

EXAMPLES = {
    "Доступ к порталу": "Пароль от кабинета ввожу верно но всплывает сообщение о блокировке",
    "Сбой печати": "В очереди печати висит первое задание и блокирует остальные",
    "Неясное обращение": "Помогите скорее ничего не получается",
    "Две проблемы": "Забыл пароль а ещё застряла бумага в принтере",
}
BG, WHITE, INK, MUTED, TEAL, LINE = "#edf3f4", "#ffffff", "#142d36", "#47646e", "#05665b", "#d7e2e4"
FONT = "Segoe UI"
ALL_QUEUES = "Все очереди"


class Desk:
    def __init__(self, window, model, policy, journal):
        self.window, self.model, self.policy, self.journal = window, model, policy, journal
        self.case_index = CaseIndex(model, read_split("train"))
        window.title("Маршрутизация заявок · ИИС 3.0")
        window.geometry("1080x810")
        window.minsize(940, 750)
        window.configure(bg=BG)
        style = ttk.Style(window)
        style.theme_use("clam")
        style.configure("Treeview", background=WHITE, fieldbackground=WHITE, foreground=INK,
                        rowheight=28, font=(FONT, 10))
        style.configure("Treeview.Heading", background="#e2ecee", foreground=INK, font=(FONT, 10, "bold"))
        style.map("Treeview", background=[("selected", "#c4e9e2")], foreground=[("selected", INK)])
        style.configure("TCombobox", padding=4)

        header = tk.Frame(window, bg=INK, padx=27, pady=16)
        header.pack(fill="x")
        self.label(header, "Техническая поддержка", 22, True, INK, WHITE).pack(anchor="w")
        self.label(header, "Классификация заявок и распределение по очередям", 10, False, INK, "#bed5d9").pack(anchor="w")
        content = tk.Frame(window, bg=BG, padx=22, pady=14)
        content.pack(fill="both", expand=True)
        workspace = tk.Frame(content, bg=BG)
        workspace.pack(fill="x", pady=(0, 13))
        workspace.grid_columnconfigure((0, 1), weight=1, uniform="workspace")

        intake = self.panel(workspace, 0, "Новое обращение")
        selectors = tk.Frame(intake, bg=WHITE)
        selectors.pack(fill="x", pady=(7, 8))
        self.example = ttk.Combobox(selectors, state="readonly", values=list(EXAMPLES), width=29)
        self.example.set("Выбрать пример")
        self.example.pack(side="left")
        self.example.bind("<<ComboboxSelected>>", self.use_example)
        self.description = tk.Text(intake, height=6, wrap="word", font=(FONT, 11), bg="#f5f8f9", fg=INK,
                                   insertbackground=INK, relief="flat", padx=10, pady=8, undo=True,
                                   highlightthickness=1, highlightbackground=LINE, highlightcolor=TEAL)
        self.description.pack(fill="both", expand=True)
        controls = tk.Frame(intake, bg=WHITE)
        controls.pack(fill="x", pady=(10, 0))
        self.submit_button = self.button(controls, "Распределить и сохранить", self.submit, primary=True)
        self.submit_button.pack(side="left")
        self.button(controls, "Очистить", self.clear_input).pack(side="left", padx=7)

        output = self.panel(workspace, 1, "Результат")
        self.route_text = tk.StringVar(value="Очередь появится после распределения")
        self.detail = tk.StringVar(value="Заполните описание или выберите пример.")
        tk.Label(output, textvariable=self.route_text, font=(FONT, 16, "bold"), bg=WHITE, fg=TEAL,
                 wraplength=400, justify="left", anchor="w").pack(fill="x", pady=(13, 5))
        tk.Label(output, textvariable=self.detail, font=(FONT, 10), bg=WHITE, fg=INK,
                 wraplength=400, justify="left", anchor="w").pack(fill="x")

        history = tk.Frame(content, bg=WHITE, padx=15, pady=10, highlightthickness=1, highlightbackground=LINE)
        history.pack(fill="both", expand=True)
        journal_tools = tk.Frame(history, bg=WHITE)
        journal_tools.pack(fill="x", pady=(0, 8))
        self.label(journal_tools, "Журнал заявок", 14, True).pack(side="left", padx=(0, 12))
        self.queue_filter = ttk.Combobox(journal_tools, state="readonly", width=22,
                                         values=[ALL_QUEUES, *QUEUES.values()])
        self.queue_filter.set(ALL_QUEUES)
        self.queue_filter.bind("<<ComboboxSelected>>", lambda _: self.refresh())
        self.queue_filter.pack(side="left")
        self.button(journal_tools, "Выгрузить CSV", self.export_dialog).pack(side="right")
        self.button(journal_tools, "Разбор заявки", self.inspect_ticket).pack(side="right", padx=7)
        self.button(journal_tools, "Импортировать CSV", self.import_dialog).pack(side="right", padx=7)
        table = tk.Frame(history, bg=WHITE)
        table.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table, columns=("id", "text", "initial", "current"),
                                  show="headings", selectmode="browse", height=5)
        for key, title, width, stretch in [
            ("id", "№", 45, False), ("text", "Описание", 380, True),
            ("initial", "Первоначальная очередь", 190, False), ("current", "Текущая очередь", 185, False)
        ]:
            self.table.heading(key, text=title)
            self.table.column(key, width=width, stretch=stretch)
        self.table.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.table.yview)
        scroll.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=scroll.set)
        self.table.bind("<<TreeviewSelect>>", self.select_row)
        self.table.bind("<Double-1>", self.show_ticket)

        manual = tk.Frame(history, bg=WHITE)
        manual.pack(fill="x", pady=(8, 0))
        self.selected_text = tk.StringVar(value="Выберите строку")
        tk.Label(manual, textvariable=self.selected_text, bg=WHITE, fg=MUTED, font=(FONT, 10)).pack(side="left")
        self.queue_choice = ttk.Combobox(manual, state="readonly", values=list(QUEUES.values()), width=25)
        self.queue_choice.pack(side="left", padx=8)
        self.move_button = self.button(manual, "Изменить очередь", self.move)
        self.move_button.pack(side="left")
        self.status = tk.StringVar(value="Готово к работе")
        tk.Label(content, textvariable=self.status, bg=BG, fg=MUTED, font=(FONT, 10), anchor="w").pack(fill="x", pady=(7, 0))
        self.refresh()

    def label(self, frame, text, size, bold=False, bg=WHITE, fg=INK):
        return tk.Label(frame, text=text, bg=bg, fg=fg, font=(FONT, size, "bold" if bold else "normal"))

    def panel(self, workspace, column, title):
        panel = tk.Frame(workspace, bg=WHITE, padx=15, pady=12, highlightthickness=1, highlightbackground=LINE)
        panel.grid(row=0, column=column, sticky="nsew", padx=(0, 7) if column == 0 else (7, 0))
        self.label(panel, title, 14, True).pack(anchor="w")
        return panel

    def button(self, frame, text, callback, primary=False):
        return tk.Button(frame, text=text, command=callback, bg=TEAL if primary else "#e8f1f2",
                         fg=WHITE if primary else INK, activebackground="#c8e6e0", activeforeground=INK,
                         font=(FONT, 10, "bold"), relief="flat", padx=10, pady=5, cursor="hand2")

    def clear_input(self):
        self.description.delete("1.0", "end")
        self.example.set("Выбрать пример")
        self.description.focus_set()

    def use_example(self, _event):
        self.description.delete("1.0", "end")
        self.description.insert("1.0", EXAMPLES[self.example.get()])

    def submit(self):
        text = self.description.get("1.0", "end-1c").strip()
        try:
            validate_text(text)
            prediction = predict(self.model, text, explain=True)
            queue, reason = route(prediction, self.policy)
            record_id = self.journal.add(text, prediction, queue)
        except (ValueError, sqlite3.Error) as error:
            self.status.set(str(error))
            return
        top = CATEGORIES[prediction.category][0]
        second = CATEGORIES[prediction.second_category][0]
        hints = ", ".join(prediction.signals) or "нет положительных вкладов"
        self.route_text.set(f"№{record_id} · {QUEUES[queue]}")
        self.detail.set(f"Первая: {top} ({prediction.score:.1%})\n"
                        f"Вторая: {second} ({prediction.second_score:.1%})\n"
                        f"{reason}\n"
                        f"Признаки: {hints}\n"
                        f"Метод ключевых слов: {QUEUES[baseline(text)]}")
        self.queue_filter.set(ALL_QUEUES)
        self.refresh(record_id)
        self.status.set(f"Заявка №{record_id} сохранена.")

    def refresh(self, record_id=None):
        reverse = {name: code for code, name in QUEUES.items()}
        matching = reverse.get(self.queue_filter.get())
        self.records = {str(row["id"]): row for row in self.journal.recent(matching)}
        for item in self.table.get_children():
            self.table.delete(item)
        for key, row in self.records.items():
            preview = " ".join(row["text"].split())
            if len(preview) > 57:
                preview = preview[:56] + "…"
            self.table.insert("", "end", iid=key, values=(key, preview, QUEUES[row["initial_queue"]],
                                                          QUEUES[row["current_queue"]]))
        self.selected_text.set("Выберите строку")
        self.queue_choice.set("")
        if str(record_id) in self.records:
            self.table.selection_set(str(record_id))
            self.table.see(str(record_id))
            self.select_row()

    def import_dialog(self):
        path = filedialog.askopenfilename(title="Импорт заявок", filetypes=[("CSV (UTF-8)", "*.csv")],
                                          parent=self.window)
        if path:
            self.import_csv(path)

    def import_csv(self, path):
        try:
            result = import_file(path, self.model, self.policy, self.journal)
        except (OSError, ValueError, sqlite3.Error) as error:
            self.route_text.set("Импорт отклонён")
            self.detail.set(str(error))
            self.status.set("Изменений в журнале нет.")
            return
        self.route_text.set(f"Пакет №{result.batch_id}")
        self.detail.set(describe(result))
        self.queue_filter.set(ALL_QUEUES)
        self.refresh()
        self.status.set("Импорт завершён." if result.created else "Повторный импорт не создал заявок.")

    def export_dialog(self):
        path = filedialog.asksaveasfilename(title="Выгрузка журнала", defaultextension=".csv",
                                            filetypes=[("CSV (UTF-8)", "*.csv")], parent=self.window)
        if not path:
            return
        try:
            self.journal.export(path)
        except (OSError, ValueError, sqlite3.Error) as error:
            self.status.set(str(error))
            return
        self.status.set(f"Журнал выгружен: {path}")

    def select_row(self, _event=None):
        selected = self.table.selection()
        if selected:
            row = self.records[selected[0]]
            self.selected_text.set(f"Заявка №{row['id']}")
            self.queue_choice.set(QUEUES[row["current_queue"]])

    def show_ticket(self, _event):
        selected = self.table.selection()
        if selected:
            row = self.records[selected[0]]
            messagebox.showinfo(f"Заявка №{row['id']}", row["text"], parent=self.window)

    def inspect_ticket(self):
        selected = self.table.selection()
        if not selected:
            self.status.set("Выберите заявку для разбора.")
            return
        row = self.records[selected[0]]
        prediction = predict(self.model, row["text"], explain=True)
        similar = self.case_index.search(row["text"])
        detail = tk.Toplevel(self.window)
        detail.title(f"Разбор заявки №{row['id']} · ИИС 3.0")
        detail.geometry("1020x780")
        detail.configure(bg=BG)
        self.label(detail, f"Заявка №{row['id']} · {QUEUES[row['current_queue']]}", 19, True,
                   BG).pack(anchor="w", padx=22, pady=(19, 9))
        text = tk.Text(detail, wrap="word", font=(FONT, 11), padx=18, pady=15,
                       bg=WHITE, fg=INK, relief="flat")
        text.pack(fill="both", expand=True, padx=22, pady=(0, 20))
        text.insert("1.0", row["text"] + "\n\n" +
                    operator_brief(prediction, row["current_queue"], similar))
        text.configure(state="disabled")

    def move(self):
        selected = self.table.selection()
        reverse = {name: code for code, name in QUEUES.items()}
        queue = reverse.get(self.queue_choice.get())
        if not selected or queue is None:
            self.status.set("Выберите заявку и нужную очередь.")
            return
        record_id = int(selected[0])
        try:
            changed = self.journal.move(record_id, queue)
        except (ValueError, sqlite3.Error) as error:
            self.status.set(str(error))
            return
        self.refresh(record_id)
        self.status.set(f"Очередь заявки №{record_id} изменена." if changed else
                        f"Заявка №{record_id} уже находится в выбранной очереди.")


def main():
    window = tk.Tk()
    Desk(window, fit_model(read_split("train")), load_policy(), Journal())
    window.mainloop()


if __name__ == "__main__":
    main()

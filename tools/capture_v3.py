"""Рабочее окно новой функции для съёмки через Computer Use."""
import sys,tempfile
from pathlib import Path
from tkinter import Tk
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import Desk,EXAMPLES
from journal import Journal
from triage import fit_model,load_policy,read_split
with tempfile.TemporaryDirectory() as folder:
    root=Tk()
    desk=Desk(root,fit_model(read_split("train")),load_policy(),Journal(Path(folder)/"capture.sqlite3"))
    root.title("ИИС 3.0 · Рабочий сценарий разбора")
    root.geometry("1280x850+40+35")
    root.attributes("-topmost",True)
    desk.example.set("Неясное обращение");desk.use_example(None);desk.submit()
    def inspect(_event=None):
        desk.inspect_ticket()
        for w in root.winfo_children():
            if "toplevel" in str(w):
                w.geometry("1100x780+95+45")
                w.attributes("-topmost",True)
    root.bind("<F8>",inspect)
    root.mainloop()

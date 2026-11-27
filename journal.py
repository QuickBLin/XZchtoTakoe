"""Локальный журнал. Первоначальный маршрут хранится отдельно от исправлений."""
import csv
import sqlite3
from tempfile import NamedTemporaryFile
from contextlib import closing
from pathlib import Path
from datetime import datetime

from triage import QUEUES, ROOT

DB = ROOT / "instance" / "tickets.sqlite3"


class Journal:
    def __init__(self, path=DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY,
                    created TEXT NOT NULL,
                    text TEXT NOT NULL,
                    predicted TEXT NOT NULL,
                    score REAL NOT NULL,
                    initial_queue TEXT NOT NULL,
                    current_queue TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS changes (
                    id INTEGER PRIMARY KEY,
                    ticket_id INTEGER NOT NULL REFERENCES tickets(id),
                    changed TEXT NOT NULL,
                    previous_queue TEXT NOT NULL,
                    new_queue TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batches (
                    id INTEGER PRIMARY KEY, digest TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL, imported TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_items (
                    batch_id INTEGER NOT NULL REFERENCES batches(id),
                    external_id TEXT NOT NULL,
                    ticket_id INTEGER NOT NULL UNIQUE REFERENCES tickets(id),
                    PRIMARY KEY (batch_id, external_id)
                );
            """)

    def connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def add(self, text, prediction, queue):
        with closing(self.connect()) as db, db:
            return self.insert(db, text, prediction, queue)

    @staticmethod
    def insert(db, text, prediction, queue):
        cursor = db.execute(
            "INSERT INTO tickets(created,text,predicted,score,initial_queue,current_queue) VALUES(?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), text,
             prediction.category, prediction.score, queue, queue),
        )
        return cursor.lastrowid

    def add_batch(self, digest, filename, decisions):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT id FROM batches WHERE digest = ?", (digest,)).fetchone()
            if existing is not None:
                batch_id = existing["id"]
                queues = db.execute("""
                    SELECT t.initial_queue FROM batch_items b JOIN tickets t ON t.id = b.ticket_id
                    WHERE b.batch_id = ? ORDER BY t.id
                """, (batch_id,)).fetchall()
                return batch_id, False, [row["initial_queue"] for row in queues]
            batch_id = db.execute(
                "INSERT INTO batches(digest, filename, imported) VALUES(?,?,?)",
                (digest, filename, datetime.now().isoformat(timespec="seconds")),
            ).lastrowid
            for ref, text, prediction, queue in decisions:
                ticket_id = self.insert(db, text, prediction, queue)
                db.execute("INSERT INTO batch_items VALUES(?,?,?)", (batch_id, ref, ticket_id))
            return batch_id, True, [queue for _, _, _, queue in decisions]

    def recent(self, queue=None):
        with closing(self.connect()) as db:
            statement = "SELECT * FROM tickets"
            params = ()
            if queue is not None:
                statement += " WHERE current_queue = ?"
                params = (queue,)
            return db.execute(statement + " ORDER BY id DESC LIMIT 100", params).fetchall()

    def export(self, path):
        path = Path(path)
        if path.resolve() == self.path.resolve():
            raise ValueError("Укажите отдельный CSV вместо файла базы данных.")
        temp_path = None
        try:
            with closing(self.connect()) as db, NamedTemporaryFile(
                mode="w", delete=False, dir=path.parent, suffix=".csv",
                encoding="utf-8-sig", newline="",
            ) as file:
                temp_path = Path(file.name)
                rows = db.execute("""
                    SELECT t.id,t.created,b.batch_id,b.external_id,t.text,t.predicted,t.score,
                           t.initial_queue,t.current_queue
                    FROM tickets t LEFT JOIN batch_items b ON t.id=b.ticket_id ORDER BY t.id
                """)
                writer = csv.writer(file)
                writer.writerow([column[0] for column in rows.description])
                for row in rows:
                    writer.writerow([
                        "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
                        else value for value in row
                    ])
            temp_path.replace(path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def move(self, record_id, new_queue):
        if new_queue not in QUEUES:
            raise ValueError("Выберите очередь из списка.")
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT current_queue FROM tickets WHERE id = ?", (record_id,)).fetchone()
            if row is None:
                raise ValueError("Заявка отсутствует в журнале.")
            previous = row["current_queue"]
            if previous == new_queue:
                return False
            db.execute("UPDATE tickets SET current_queue = ? WHERE id = ?", (new_queue, record_id))
            db.execute(
                "INSERT INTO changes(ticket_id,changed,previous_queue,new_queue) VALUES(?,?,?,?)",
                (record_id, datetime.now().isoformat(timespec="seconds"), previous, new_queue),
            )
            return True

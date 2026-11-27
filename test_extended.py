"""Регрессия импорта, журнала и операторского окна."""
import csv
import shutil
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from tkinter import Tk

from app import Desk, EXAMPLES
from batch import import_file, parse_csv
from evaluate import check_data
from journal import Journal
from triage import QUEUES, REVIEW, ROOT, fit_model, load_policy, predict, read_split

EXAMPLES_DIR = ROOT / "examples"


class CsvTests(unittest.TestCase):
    def test_utf8_bom_and_multiline(self):
        raw = '\ufeffid,text\nA,"Строка, с запятой\nпродолжение"\n'.encode("utf-8")
        self.assertEqual(parse_csv(raw), [("A", "Строка, с запятой\nпродолжение")])

    def test_wrong_encoding(self):
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            parse_csv(b"id,text\na,\xff\n")

    def test_wrong_header(self):
        with self.assertRaisesRegex(ValueError, "id,text"):
            parse_csv(b"number,message\n1,hello\n")

    def test_wrong_column_count(self):
        with self.assertRaisesRegex(ValueError, "Строка 2"):
            parse_csv(b"id,text\na,hello,extra\n")

    def test_duplicate_external_id(self):
        with self.assertRaisesRegex(ValueError, "повторён"):
            parse_csv(b"id,text\na,hello\na,hello again\n")

    def test_empty_batch(self):
        with self.assertRaisesRegex(ValueError, "хотя бы одно"):
            parse_csv(b"id,text\n")

    def test_file_and_row_limits(self):
        with self.assertRaisesRegex(ValueError, "512"):
            parse_csv(b" " * (512 * 1024 + 1))
        raw = "id,text\n" + "".join(f"{i},help\n" for i in range(501))
        with self.assertRaisesRegex(ValueError, "500"):
            parse_csv(raw.encode())

    def test_split_integrity(self):
        splits = {name: read_split(name) for name in ("train", "validation", "test", "stress")}
        check_data(splits)
        self.assertEqual([len(rows) for rows in splits.values()], [144, 36, 60, 72])


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fit_model(read_split("train"))
        cls.policy = load_policy()

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.journal = Journal(self.folder / "test.sqlite3")
        self.demo = EXAMPLES_DIR / "demo_batch.csv"
        self.rejected = EXAMPLES_DIR / "rejected_batch.csv"
        self.known = predict(self.model, EXAMPLES["Доступ к порталу"], explain=True)

    def import_one(self, path):
        return import_file(path, self.model, self.policy, self.journal)

    def test_valid_batch_and_queue_totals(self):
        result = self.import_one(self.demo)
        self.assertEqual(result.count, 8)
        self.assertEqual(result.by_queue[REVIEW], 1)
        self.assertEqual(result.by_queue["print"], 2)
        self.assertEqual(len(self.journal.recent()), 8)

    def test_invalid_batch_leaves_journal_unchanged(self):
        self.import_one(self.demo)
        with self.assertRaisesRegex(ValueError, "Строка 3"):
            self.import_one(self.rejected)
        self.assertEqual(len(self.journal.recent()), 8)

    def test_renamed_duplicate_is_idempotent(self):
        first = self.import_one(self.demo)
        other_name = self.folder / "renamed.csv"
        shutil.copyfile(self.demo, other_name)
        same = self.import_one(other_name)
        self.assertEqual(same.batch_id, first.batch_id)
        self.assertFalse(same.created)
        self.assertEqual(len(self.journal.recent()), 8)

    def test_concurrent_retry_is_idempotent(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = list(executor.map(self.import_one, [self.demo, self.demo]))
        self.assertEqual(first.batch_id, second.batch_id)
        self.assertEqual(int(first.created) + int(second.created), 1)
        self.assertEqual(len(self.journal.recent()), 8)

    def test_sql_failure_rolls_back_entire_batch(self):
        proposals = [("duplicate", "Заявка А", self.known, "access"),
                     ("duplicate", "Заявка Б", self.known, "access")]
        with self.assertRaises(sqlite3.IntegrityError):
            self.journal.add_batch("test-digest", "bad.csv", proposals)
        self.assertEqual(self.journal.recent(), [])
        with closing(self.journal.connect()) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM batches").fetchone()[0], 0)

    def test_export_contains_initial_and_current(self):
        self.import_one(self.demo)
        self.journal.move(8, REVIEW)
        out = self.folder / "journal.csv"
        self.journal.export(out)
        with out.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 8)
        self.assertEqual((rows[-1]["external_id"], rows[-1]["initial_queue"], rows[-1]["current_queue"]),
                         ("DEMO-08", "print", REVIEW))

    def test_export_retains_literal_formula_like_text(self):
        self.journal.add("=Первая строка", self.known, "access")
        out = self.folder / "export.csv"
        self.journal.export(out)
        with out.open(encoding="utf-8-sig", newline="") as file:
            row = next(csv.DictReader(file))
        self.assertEqual(row["text"], "'=Первая строка")
        self.assertEqual(self.journal.recent()[0]["text"], "=Первая строка")

    def test_guard_prevents_export_over_database(self):
        with self.assertRaisesRegex(ValueError, "отдельный CSV"):
            self.journal.export(self.journal.path)
        self.assertEqual(self.journal.recent(), [])

    def test_filter_uses_current_queue(self):
        self.import_one(self.demo)
        self.journal.move(8, REVIEW)
        self.assertEqual([row["id"] for row in self.journal.recent(REVIEW)], [8, 7])

    def test_original_database_reads_with_new_tables(self):
        old_path = self.folder / "v1.sqlite3"
        with closing(sqlite3.connect(old_path)) as db, db:
            db.executescript("""
                CREATE TABLE tickets(id INTEGER PRIMARY KEY, created TEXT NOT NULL, text TEXT NOT NULL,
                    predicted TEXT NOT NULL, score REAL NOT NULL, initial_queue TEXT NOT NULL,
                    current_queue TEXT NOT NULL);
                INSERT INTO tickets VALUES (1,'2026-11-26','Старое обращение','access',0.9,'access','access');
            """)
        upgraded = Journal(old_path)
        self.assertEqual(upgraded.recent()[0]["text"], "Старое обращение")
        self.assertEqual(import_file(self.demo, self.model, self.policy, upgraded).count, 8)
        self.assertEqual(len(upgraded.recent()), 9)

    def test_window_full_batch_workflow(self):
        root = Tk()
        root.withdraw()
        try:
            desk = Desk(root, self.model, self.policy, self.journal)
            desk.import_csv(self.demo)
            root.update()
            self.assertIn("Пакет №1", desk.route_text.get())
            self.assertEqual(len(desk.table.get_children()), 8)
            desk.import_csv(self.rejected)
            self.assertIn("отклонён", desk.route_text.get())
            self.assertEqual(len(desk.table.get_children()), 8)
            desk.queue_filter.set(QUEUES[REVIEW])
            desk.refresh()
            self.assertEqual(desk.table.get_children(), ("7",))
            desk.table.selection_set("7")
            desk.select_row()
            desk.queue_choice.set(QUEUES["access"])
            desk.move_button.invoke()
            root.update()
            self.assertEqual(desk.table.get_children(), ())
            self.assertEqual(Journal(self.journal.path).recent("access")[0]["id"], 7)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)

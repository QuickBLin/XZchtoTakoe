"""Проверки модели, SQLite и сценария в настоящем окне Tk."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from tkinter import Tk

from app import Desk, EXAMPLES
from journal import Journal
from triage import QUEUES, REVIEW, ROOT, baseline, fit_model, load_policy, predict, read_split, route, validate_text


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fit_model(read_split("train"))
        cls.policy = load_policy()

    def test_baseline(self):
        self.assertEqual(baseline("Заблокирована учётная запись, забыл пароль"), "access")
        self.assertEqual(baseline("Пароль и принтер"), REVIEW)
        self.assertEqual(baseline("Срочно!"), REVIEW)

    def test_prediction_and_route(self):
        result = predict(self.model, EXAMPLES["Доступ к порталу"], explain=True)
        self.assertEqual(result.category, "access")
        self.assertEqual(route(result, self.policy)[0], "access")
        self.assertTrue(result.signals)
        unknown = predict(self.model, EXAMPLES["Неясное обращение"])
        self.assertEqual(route(unknown, self.policy)[0], REVIEW)

    def test_invalid_input(self):
        for text in (" ", "123!", "а" * 2001):
            with self.subTest(text=text[:10]), self.assertRaises(ValueError):
                validate_text(text)

    def test_route_boundaries(self):
        from triage import Prediction
        policy = {"min_score": 0.50, "min_gap": 0.25}
        self.assertEqual(route(Prediction("access", 0.50, "network", 0.25, []), policy)[0], "access")
        self.assertEqual(route(Prediction("access", 0.50, "network", 0.26, []), policy)[0], REVIEW)
        self.assertEqual(route(Prediction("access", 0.49, "network", 0.20, []), policy)[0], REVIEW)

    def test_window_and_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.sqlite3"
            journal = Journal(path)
            root = Tk()
            root.withdraw()
            try:
                desk = Desk(root, self.model, self.policy, journal)
                desk.submit()
                self.assertIn("хотя бы одну букву", desk.status.get())
                self.assertEqual(len(journal.recent()), 0)

                for name, expected in (("Доступ к порталу", "access"), ("Неясное обращение", REVIEW)):
                    desk.example.set(name)
                    desk.use_example(None)
                    desk.submit()
                    root.update()
                    self.assertIn(QUEUES[expected], desk.route_text.get())
                    self.assertEqual(journal.recent()[0]["current_queue"], expected)

                record_id = journal.recent()[0]["id"]
                desk.table.selection_set(str(record_id))
                desk.select_row()
                desk.queue_choice.set(QUEUES["network"])
                desk.move()
                root.update()
                reopened = Journal(path).recent()[0]
                self.assertEqual(reopened["initial_queue"], REVIEW)
                self.assertEqual(reopened["current_queue"], "network")
                self.assertFalse(journal.move(record_id, "network"))
                with closing(sqlite3.connect(path)) as db:
                    changes = db.execute("SELECT previous_queue, new_queue FROM changes").fetchall()
                self.assertEqual(changes, [(REVIEW, "network")])
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)

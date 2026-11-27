import tempfile
import unittest
from pathlib import Path
from tkinter import Tk,Toplevel

from app import Desk,EXAMPLES
from assistant import CaseIndex,GUIDANCE,operator_brief
from journal import Journal
from triage import CATEGORIES,REVIEW,fit_model,load_policy,predict,read_split


class AssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=read_split("train")
        cls.model=fit_model(cls.rows)
        cls.index=CaseIndex(cls.model,cls.rows)

    def test_exact_text_has_maximum_similarity(self):
        source=self.rows[0]
        result=self.index.search(source["text"])
        self.assertEqual(result[0].identifier,source["id"])
        self.assertAlmostEqual(result[0].similarity,1)
        self.assertEqual(len(result),3)

    def test_search_uses_training_ids_only(self):
        result=self.index.search("Принтер жует бумагу")
        valid={row["id"] for row in self.rows}
        self.assertTrue(result)
        self.assertTrue(all(item.identifier in valid for item in result))
        self.assertTrue(all(0<item.similarity<=1.00000001 for item in result))

    def test_empty_and_zero_feature_search(self):
        self.assertEqual(self.index.search(""),[])
        self.assertEqual(self.index.search("qqqqzzzzxxxxxxxx"),[])
        self.assertEqual(self.index.search("принтер",limit=0),[])

    def test_deterministic_search(self):
        self.assertEqual(self.index.search("Печать перестала работать"),
                         self.index.search("Печать перестала работать"))

    def test_guidance_covers_all_routes(self):
        self.assertEqual(set(GUIDANCE),set(CATEGORIES)|{REVIEW})
        for action,question in GUIDANCE.values():
            self.assertTrue(action)
            self.assertTrue(question.endswith("?"))

    def test_review_brief_asks_for_clarification(self):
        prediction=predict(self.model,"Помогите скорее ничего не получается")
        brief=operator_brief(prediction,REVIEW,[])
        self.assertIn("нескольких независимых",brief)
        self.assertIn("не найдено",brief)

    def test_gui_inspection_opens_and_keeps_journal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Tk();root.withdraw()
            journal=Journal(Path(folder)/"test.sqlite3")
            try:
                desk=Desk(root,self.model,load_policy(),journal)
                desk.inspect_ticket()
                self.assertIn("Выберите заявку",desk.status.get())
                desk.example.set("Доступ к порталу");desk.use_example(None);desk.submit()
                desk.inspect_ticket();root.update()
                dialogs=[w for w in root.winfo_children() if isinstance(w,Toplevel)]
                self.assertEqual(len(dialogs),1)
                self.assertIn("Разбор заявки №1",dialogs[0].title())
                self.assertEqual(len(journal.recent()),1)
            finally:
                root.destroy()


if __name__=="__main__":
    unittest.main(verbosity=2)

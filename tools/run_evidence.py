"""Повторяемый прогон с журналами команд, кодами выхода и экспортом примера."""
import os
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with tempfile.TemporaryDirectory() as name:
        db = str(Path(name) / "demo.sqlite3")
        jobs = [
            ("system", ["-m", "unittest", "-v", "test_system", "test_extended", "test_assistant"], 0),
            ("experiments", ["experiment.py"], 0),
            ("research", ["research.py"], 0),
            ("batch_first", ["route_batch.py", "examples/demo_batch.csv", "--db", db,
                             "--export", str(RESULTS / "demo_export.csv")], 0),
            ("batch_repeat", ["route_batch.py", "examples/demo_batch.csv", "--db", db], 0),
            ("batch_reject", ["route_batch.py", "examples/rejected_batch.csv", "--db", db], 2),
        ]
        records = {}
        for label, args, expected in jobs:
            process = subprocess.run([sys.executable, *args], env=env, cwd=ROOT, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, encoding="utf-8", check=False)
            command = "python " + " ".join(args)
            observed = process.stdout
            command = command.replace(db, "TEMP/demo.sqlite3").replace(str(RESULTS), "results")
            observed = observed.replace(str(RESULTS), "results").replace(db, "TEMP/demo.sqlite3")
            log = f"> {command}\n\n{observed}\nКод выхода: {process.returncode}. Ожидался: {expected}.\n"
            (RESULTS / f"run_{label}.log").write_text(log, encoding="utf-8")
            print(f"{label:15} exit={process.returncode}, expected={expected}")
            records[label] = {"command": command, "exit_code": process.returncode, "expected": expected}
        (RESULTS / "run_manifest.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return int(any(row["exit_code"] != row["expected"] for row in records.values()))


if __name__ == "__main__":
    raise SystemExit(main())

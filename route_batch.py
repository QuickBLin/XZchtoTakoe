"""CLI для пакетного импорта и экспорта журнала."""
import argparse
import sqlite3
from pathlib import Path

from batch import describe, import_file
from journal import Journal
from triage import fit_model, load_policy, read_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, help="CSV с заголовком id,text")
    parser.add_argument("--db", type=Path, help="Отдельный журнал SQLite")
    parser.add_argument("--export", type=Path, help="Путь к выгрузке текущего журнала")
    args = parser.parse_args()
    if not (args.csv or args.export):
        parser.error("Укажите входной CSV или --export.")
    if args.csv and args.export and args.csv.resolve() == args.export.resolve():
        parser.error("Входной CSV и выгрузка должны быть разными файлами.")
    try:
        journal = Journal(args.db) if args.db else Journal()
        if args.csv:
            model = fit_model(read_split("train"))
            print(describe(import_file(args.csv, model, load_policy(), journal)))
        if args.export:
            journal.export(args.export)
            print(f"Журнал сохранён: {args.export}")
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(2, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()

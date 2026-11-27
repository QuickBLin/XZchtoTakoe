"""Импорт CSV в журнал. Одинаковое содержимое повторно не импортируется."""
import csv
import hashlib
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from triage import QUEUES, REVIEW, predict, route, validate_text

MAX_BYTES = 512 * 1024
MAX_ROWS = 500


@dataclass(frozen=True)
class BatchResult:
    batch_id: int
    created: bool
    count: int
    by_queue: dict


def parse_csv(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError("CSV должен быть не больше 512 КиБ.")
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Сохраните CSV в кодировке UTF-8.") from error

    try:
        reader = csv.reader(io.StringIO(decoded, newline=""), strict=True)
        if next(reader, []) != ["id", "text"]:
            raise ValueError("Первая строка CSV должна содержать id,text.")
        rows, ids = [], set()
        for line in reader:
            if len(rows) == MAX_ROWS:
                raise ValueError("CSV должен содержать до 500 обращений.")
            if len(line) != 2:
                raise ValueError(f"Строка {reader.line_num}: ожидаются два столбца.")
            ref, text = (cell.strip() for cell in line)
            if not ref or len(ref) > 80 or ref in ids:
                raise ValueError(f"Строка {reader.line_num}: ID пуст, длиннее 80 символов или повторён.")
            try:
                validate_text(text)
            except ValueError as error:
                raise ValueError(f"Строка {reader.line_num}: {error}") from error
            rows.append((ref, text))
            ids.add(ref)
    except csv.Error as error:
        raise ValueError(f"Строка {reader.line_num}: нарушен формат CSV.") from error
    if not rows:
        raise ValueError("Добавьте хотя бы одно обращение после заголовка.")
    return rows


def import_file(path, model, policy, journal):
    path = Path(path)
    with path.open("rb") as file:
        raw = file.read(MAX_BYTES + 1)
    rows = parse_csv(raw)
    digest = hashlib.sha256(raw).hexdigest()
    decisions = []
    for ref, text in rows:
        prediction = predict(model, text)
        queue = route(prediction, policy)[0]
        decisions.append((ref, text, prediction, queue))
    batch_id, created, original_queues = journal.add_batch(digest, path.name, decisions)
    return BatchResult(batch_id, created, len(original_queues), dict(Counter(original_queues)))


def describe(result):
    event = "Импорт выполнен" if result.created else "Этот файл уже был импортирован"
    manual = result.by_queue.get(REVIEW, 0)
    details = ", ".join(f"{QUEUES[code]} — {qty}" for code, qty in result.by_queue.items())
    return (f"{event}. Пакет №{result.batch_id}, строк: {result.count}.\n"
            f"Автоматически: {result.count - manual}, вручную: {manual}.\n{details}")

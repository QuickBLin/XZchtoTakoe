"""Данные, классификаторы и правила распределения."""
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent
REVIEW = "review"
CATEGORIES = {
    "access": ("Доступ и учётные записи", "Управление доступом"),
    "network": ("Сеть и VPN", "Сетевые сервисы"),
    "software": ("Программы и лицензии", "Поддержка ПО"),
    "hardware": ("Компьютеры и периферия", "Обслуживание техники"),
    "mail": ("Почта и календарь", "Почтовые сервисы"),
    "print": ("Печать и сканирование", "Сервис печати"),
}
QUEUES = {code: details[1] for code, details in CATEGORIES.items()}
QUEUES[REVIEW] = "Ручная проверка"
MAX_LENGTH = 2000

# Список слов и устойчивых фраз подготовлен по описаниям классов и train.
KEYWORDS = {
    "access": ("пароль", "пароля", "логин", "учётная запись", "учётной записи", "доступ", "прав", "заблокирована", "авторизация", "двухфакторная", "роль"),
    "network": ("vpn", "wi-fi", "wi fi", "интернет", "интернета", "сеть", "сети", "dns", "ip", "роутер", "сетевой порт", "шлюза"),
    "software": ("программа", "программы", "приложение", "приложения", "обновление", "обновления", "установка", "установить", "лицензия", "лицензию", "редактор"),
    "hardware": ("ноутбук", "монитор", "клавиатура", "мышь", "батарея", "зарядка", "вентилятор", "гарнитура", "экран", "разъём"),
    "mail": ("почта", "почту", "почты", "письмо", "письма", "ящик", "ящика", "спам", "календарь", "вложения", "рассылки"),
    "print": ("принтер", "принтера", "печать", "печати", "бумага", "картридж", "тонер", "сканер", "сканирование", "мфу", "распечатка"),
}


def normalize(text):
    """Одинаковая подготовка текста при обучении и при прогнозе."""
    return " ".join(re.findall(r"[a-zа-я0-9]+", text.lower().replace("ё", "е")))


def baseline(text):
    prepared = f" {normalize(text)} "
    scores = {
        code: sum(f" {normalize(word)} " in prepared for word in words)
        for code, words in KEYWORDS.items()
    }
    best = max(scores.values())
    winners = [code for code, count in scores.items() if count == best]
    return winners[0] if best and len(winners) == 1 else REVIEW


def read_split(name):
    with (ROOT / "data" / f"{name}.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def fit_model(rows, variant="combined"):
    # Оба векторизатора изучают словарь и частоты на train.
    vectorizers = [
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
    ]
    if variant not in {"word", "char", "combined"}:
        raise ValueError(f"Неизвестный вариант признаков: {variant}")
    features = FeatureUnion([(name, vectorizer) for name, vectorizer in vectorizers
                            if variant == "combined" or variant == name])
    model = Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(C=4, max_iter=1000, solver="lbfgs")),
    ])
    model.fit([normalize(row["text"]) for row in rows], [row["label"] for row in rows])
    return model


@dataclass(frozen=True)
class Prediction:
    category: str
    score: float
    second_category: str
    second_score: float
    signals: list[str]

    @property
    def gap(self):
        return self.score - self.second_score


def predict(model, text, explain=False):
    text = normalize(text)
    # Преобразуем текст теми же векторизаторами, которые обучались на train.
    features = model.named_steps["features"]
    vector = features.transform([text])
    classifier = model.named_steps["classifier"]
    scores = classifier.predict_proba(vector)[0]
    first, second = sorted(range(len(scores)), key=lambda i: -scores[i])[:2]
    signals = []
    if explain:
        names = features.get_feature_names_out()
        # Вклад фрагмента в разницу линейных оценок лидера и второго класса.
        weights = classifier.coef_[first] - classifier.coef_[second]
        contributions = [(float(vector[0, i] * weights[i]), names[i]) for i in vector.indices]
        signals = [
            f"{name.split('__', 1)[1].strip()} ({'слово или фраза' if name.startswith('word__') else 'часть слова'})"
            for value, name in sorted(contributions, reverse=True) if value > 0
        ][:4]
    return Prediction(str(classifier.classes_[first]), float(scores[first]),
                      str(classifier.classes_[second]), float(scores[second]), signals)


def route(prediction, policy):
    if prediction.score < policy["min_score"]:
        return REVIEW, "Оценка первой категории ниже порога."
    if prediction.gap < policy["min_gap"]:
        return REVIEW, "Две первые категории получили близкие оценки."
    return prediction.category, "Оценка и отрыв прошли оба порога."


def load_policy():
    return json.loads((ROOT / "results" / "policy.json").read_text(encoding="utf-8"))


def validate_text(text):
    if len(text) > MAX_LENGTH:
        raise ValueError(f"Описание должно содержать до {MAX_LENGTH} символов.")
    if not any(letter.isalpha() for letter in normalize(text)):
        raise ValueError("Введите текст обращения, содержащий хотя бы одну букву.")

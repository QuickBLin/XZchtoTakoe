"""Предметные подсказки оператору и поиск близких обучающих примеров."""
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from triage import CATEGORIES, REVIEW, normalize

GUIDANCE = {
    "access": (
        "Уточнить этап входа и точный текст ошибки. Проверить роль пользователя по регламенту организации.",
        "Какой раздел недоступен и что написано в сообщении об ошибке?",
    ),
    "network": (
        "Уточнить способ подключения, круг затронутых устройств и доступность других ресурсов.",
        "Проблема возникает по Wi-Fi, кабелю или VPN? Есть ли она у коллег?",
    ),
    "software": (
        "Записать название и версию программы, шаги воспроизведения и последнее изменение.",
        "На каком действии программа останавливается и было ли обновление?",
    ),
    "hardware": (
        "Уточнить устройство, индикаторы, питание и наличие механического повреждения.",
        "Включается ли устройство и какие индикаторы горят?",
    ),
    "mail": (
        "Уточнить направление доставки и домен получателя, запросить текст уведомления об ошибке.",
        "Не отправляются или не приходят письма? Это происходит со всеми адресатами?",
    ),
    "print": (
        "Уточнить аппарат, состояние очереди печати, бумагу и сообщение на дисплее.",
        "Задание осталось в очереди или лист вышел с дефектом?",
    ),
    REVIEW: (
        "Уточнить объект и симптом. При нескольких независимых проблемах оформить отдельные обращения.",
        "Что именно не работает, что ожидалось и сколько разных проблем описано?",
    ),
}


@dataclass(frozen=True)
class SimilarCase:
    identifier: str
    text: str
    category: str
    similarity: float


class CaseIndex:
    def __init__(self, model, rows):
        self.features = model.named_steps["features"]
        self.rows = list(rows)
        self.vectors = self.features.transform([normalize(row["text"]) for row in self.rows])

    def search(self, text, limit=3):
        if limit <= 0 or not self.rows:
            return []
        vector = self.features.transform([normalize(text)])
        scores = cosine_similarity(vector, self.vectors)[0]
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], self.rows[index]["id"]))
        return [
            SimilarCase(self.rows[index]["id"], self.rows[index]["text"],
                        self.rows[index]["label"], float(scores[index]))
            for index in order[:limit] if scores[index] > 0
        ]


def operator_brief(prediction, queue, similar):
    action, question = GUIDANCE[queue]
    first = CATEGORIES[prediction.category][0]
    second = CATEGORIES[prediction.second_category][0]
    lines = [
        f"Предполагаемая тема — {first} ({prediction.score:.1%}).",
        f"Вторая тема — {second} ({prediction.second_score:.1%}). Отрыв {prediction.gap:.1%}.",
        "", "Следующее действие", action, "", "Вопрос пользователю", question,
        "", "Близкие примеры из train",
    ]
    for case in similar:
        lines.append(f"{case.identifier} · {CATEGORIES[case.category][0]} · сходство {case.similarity:.3f}")
        lines.append(case.text)
        lines.append("")
    if not similar:
        lines.append("Совпадающих признаков с учебными примерами не найдено.")
    lines.extend(["Сходство показывает близость признаков текста. Для назначения используется классификатор.",
                  "Вопросы и действия заданы в учебной базе знаний и требуют согласования с регламентом службы."])
    return "\n".join(lines)

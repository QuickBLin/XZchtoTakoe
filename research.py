"""Дополнительное исследование зафиксированной модели без подстройки по тестам."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from triage import CATEGORIES, ROOT, REVIEW, fit_model, load_policy, normalize, predict, read_split, route


def read_predictions(name):
    with (ROOT / "results" / f"extended_{name}.csv").open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def bootstrap(rows):
    # Парный bootstrap. В каждом повторе оба метода видят одни и те же выбранные индексы.
    rng = np.random.default_rng(2026)
    n = len(rows)
    combined = np.array([row["combined_route"] == row["label"] for row in rows], dtype=float)
    keywords = np.array([row["keywords"] == row["label"] for row in rows], dtype=float)
    indices = rng.integers(0, n, size=(5000, n))
    scores = combined[indices].mean(axis=1)
    differences = (combined-keywords)[indices].mean(axis=1)
    return {
        "repeats": 5000, "accuracy": float(combined.mean()),
        "accuracy_percentile_95": np.quantile(scores, [0.025, 0.975]).tolist(),
        "paired_difference": float((combined-keywords).mean()),
        "paired_difference_percentile_95": np.quantile(differences, [0.025, 0.975]).tolist(),
    }


def thresholds(rows):
    policies = [
        ("Без отказа", 0.0, 0.0),
        ("Мягкие", 0.25, 0.05),
        ("Рабочие", 0.35, 0.25),
        ("Строгие", 0.50, 0.35),
        ("Очень строгие", 0.70, 0.45),
    ]
    output = []
    for name, score, gap in policies:
        answers = [
            row["combined_predicted"] if float(row["combined_score"]) >= score
            and float(row["combined_gap"]) >= gap else REVIEW for row in rows
        ]
        auto = [(r, a) for r, a in zip(rows, answers) if a != REVIEW]
        correct_auto = sum(r["label"] == a for r, a in auto)
        output.append({
            "name": name, "min_score": score, "min_gap": gap,
            "auto": len(auto), "auto_correct": correct_auto, "auto_wrong": len(auto)-correct_auto,
            "manual": len(rows)-len(auto),
            "routes_correct": sum(r["label"] == a for r, a in zip(rows, answers)),
        })
    return output


def learning_curve(train):
    y = np.array([row["label"] for row in train])
    records = []
    for seed in (17, 29, 43):
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for fold, (available, holdout) in enumerate(folds.split(np.zeros(len(y)), y), 1):
            rng = np.random.default_rng(seed*100 + fold)
            by_class = {code: rng.permutation(available[y[available] == code]) for code in CATEGORIES}
            check = [train[i] for i in holdout]
            for per_class in (2, 4, 8, 16):
                selected = np.concatenate([by_class[code][:per_class] for code in CATEGORIES])
                model = fit_model([train[i] for i in selected])
                expected = [r["label"] for r in check]
                answers = model.predict([normalize(r["text"]) for r in check])
                records.append({
                    "seed": seed, "fold": fold, "train_n": len(selected), "holdout_n": len(check),
                    "macro_f1": float(f1_score(expected, answers, labels=list(CATEGORIES), average="macro")),
                    "accuracy": float(np.mean(answers == np.array(expected))),
                })
    summary = []
    for n in (12, 24, 48, 96):
        values = np.array([r["macro_f1"] for r in records if r["train_n"] == n])
        summary.append({"train_n": n, "runs": len(values), "mean_macro_f1": float(values.mean()),
                        "std_macro_f1": float(values.std(ddof=1)), "min": float(values.min()), "max": float(values.max())})
    return records, summary


def vocabulary(model):
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    names = features.get_feature_names_out()
    # Для компактного объяснения показываем только словесные признаки.
    words = [i for i, name in enumerate(names) if name.startswith("word__")]
    result = {}
    for index, code in enumerate(classifier.classes_):
        top = sorted(words, key=lambda k: -classifier.coef_[index,k])[:8]
        result[str(code)] = [{"term": names[k].split("__",1)[1], "weight": float(classifier.coef_[index,k])} for k in top]
    return {"features":len(names), "word_features":len(words), "char_features":len(names)-len(words),
            "classes": len(classifier.classes_), "top_word_features": result}


def main():
    train = read_split("train")
    curve_rows, curve = learning_curve(train)
    final = {"method": "Ретроспективная диагностика. Рабочая модель и пороги сохранены.",
             "bootstrap": {}, "thresholds": {}, "learning_curve":curve,
             "vocabulary":vocabulary(fit_model(train))}
    for split in ("test","stress"):
        rows=read_predictions(split)
        final["bootstrap"][split]=bootstrap(rows)
        final["thresholds"][split]=thresholds(rows)
    with (ROOT/"results/learning_curve.csv").open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(curve_rows[0]))
        writer.writeheader();writer.writerows(curve_rows)
    (ROOT/"results/research.json").write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"learning_curve":curve,"bootstrap":final["bootstrap"],
                      "thresholds_stress":final["thresholds"]["stress"],
                      "feature_counts":{k:v for k,v in final["vocabulary"].items() if k!="top_word_features"}},
                     ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

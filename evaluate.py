"""Настройка по validation и итоговая проверка на test."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from itertools import product

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from triage import CATEGORIES, REVIEW, ROOT, baseline, fit_model, predict, read_split, route, normalize

RESULTS = ROOT / "results"
SCORES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]
GAPS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def save_json(name, data):
    (RESULTS / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha_file(split):
    return hashlib.sha256((ROOT / "data" / f"{split}.csv").read_bytes()).hexdigest()


def check_data(splits):
    seen_text = set()
    seen_ids = set()
    for name, rows in splits.items():
        counts = Counter(row["label"] for row in rows)
        assert set(CATEGORIES).issubset(counts), f"В {name} отсутствует категория"
        assert set(counts).issubset(set(CATEGORIES) | {REVIEW}), f"Неизвестная категория в {name}"
        assert name != "train" or REVIEW not in counts, "Ручная проверка не является классом модели"
        for row in rows:
            text = normalize(row["text"])
            assert row["id"] not in seen_ids and text and text not in seen_text, row["id"]
            seen_text.add(text)
            seen_ids.add(row["id"])


def fit_policy(model, rows):
    predictions = [predict(model, row["text"]) for row in rows]
    best_rank = (-1, 0, 0)
    best_policy = None
    for score, gap in product(SCORES, GAPS):
        policy = {"min_score": score, "min_gap": gap}
        routes = [route(p, policy)[0] for p in predictions]
        auto = [(row["label"], actual) for row, actual in zip(rows, routes) if actual != REVIEW]
        if any(expected != actual for expected, actual in auto):
            continue
        # Максимум верно распределённых автоматически. При равенстве строже оба порога.
        rank = (len(auto), score, gap)
        if rank > best_rank:
            best_rank, best_policy = rank, policy
    return {**best_policy, "validation_auto_correct": best_rank[0],
            "validation_total": len(rows), "train_sha256": sha_file("train"),
            "validation_sha256": sha_file("validation")}


def classify_metrics(rows, output, label_keys):
    actual = [row["label"] for row in rows]
    p, r, f, count = precision_recall_fscore_support(actual, output, labels=label_keys, zero_division=0)
    def per_class(index):
        return {"precision": float(p[index]), "recall": float(r[index]),
                "f1": float(f[index]), "support": int(count[index])}
    return {
        "correct": sum(a == b for a, b in zip(actual, output)),
        "total": len(actual),
        "accuracy": sum(a == b for a, b in zip(actual, output)) / len(actual),
        "macro_f1": float(f1_score(actual, output, labels=label_keys, average="macro", zero_division=0)),
        "per_class": {name: per_class(index) for index, name in enumerate(label_keys)},
        "matrix_order": [*label_keys] if REVIEW in label_keys else [*label_keys, REVIEW],
        "matrix": confusion_matrix(actual, output, labels=list(label_keys) if REVIEW in label_keys else [*label_keys, REVIEW]).tolist(),
    }


def route_metrics(rows, output):
    auto = [(row["label"], actual) for row, actual in zip(rows, output) if actual != REVIEW]
    correct = sum(expected == actual for expected, actual in auto)
    review_total = sum(row["label"] == REVIEW for row in rows)
    review_held = sum(row["label"] == actual == REVIEW for row, actual in zip(rows, output))
    return {
        **classify_metrics(rows, output, [*CATEGORIES, REVIEW]),
        "auto_count": len(auto),
        "coverage": len(auto) / len(rows),
        "auto_correct": correct,
        "auto_accuracy": correct / len(auto) if auto else None,
        "review_expected": review_total,
        "review_held": review_held,
    }


def final_test(model, policy, test):
    predictions = [predict(model, row["text"]) for row in test]
    outcomes = []
    for row, prediction in zip(test, predictions):
        actual, reason = route(prediction, policy)
        outcomes.append({
            **row, "keywords": baseline(row["text"]), "ml_category": prediction.category,
            "score": round(prediction.score, 6), "second_category": prediction.second_category,
            "gap": round(prediction.gap, 6), "ml_route": actual, "route_reason": reason,
        })
    with (RESULTS / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=outcomes[0].keys())
        writer.writeheader()
        writer.writerows(outcomes)
    closed = [row for row in outcomes if row["label"] != REVIEW]
    metrics = {
        "test_sha256": sha_file("test"),
        "sample_sizes": {name: dict(Counter(row["label"] for row in read_split(name)))
                         for name in ("train", "validation", "test")},
        "classification": {
            "keywords": classify_metrics(closed, [row["keywords"] for row in closed], list(CATEGORIES)),
            "ml": classify_metrics(closed, [row["ml_category"] for row in closed], list(CATEGORIES)),
        },
        "routing": {
            "keywords": route_metrics(outcomes, [row["keywords"] for row in outcomes]),
            "ml": route_metrics(outcomes, [row["ml_route"] for row in outcomes]),
        },
    }
    save_json("metrics.json", metrics)
    for subject in ("classification", "routing"):
        for method, values in metrics[subject].items():
            print(f"{subject:14} {method:8} correct={values['correct']}/{values['total']} "
                  f"macro-F1={values['macro_f1']:.3f}  auto={values.get('auto_count', '-')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true", help="Настроить пороги без чтения test")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    splits = {name: read_split(name) for name in ("train", "validation")}
    if not args.calibrate:
        splits["test"] = read_split("test")
    check_data(splits)
    model = fit_model(splits["train"])
    if args.calibrate:
        policy = fit_policy(model, splits["validation"])
        save_json("policy.json", policy)
        print(policy)
    else:
        from triage import load_policy
        policy = load_policy()
        if any(policy[f"{n}_sha256"] != sha_file(n) for n in ("train", "validation")):
            raise ValueError("Изменились исходные данные. Сначала выполните настройку --calibrate.")
        final_test(model, policy, splits["test"])


if __name__ == "__main__":
    main()

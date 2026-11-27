"""Фиксация вариантов на validation и оценка на двух отдельных наборах."""
import argparse
import csv
import hashlib
import platform
import statistics
import sys
import time
from collections import Counter

import sklearn

from evaluate import check_data, classify_metrics, fit_policy, route_metrics, save_json, sha_file
from triage import CATEGORIES, REVIEW, ROOT, baseline, fit_model, load_policy, predict, read_split, route

VARIANTS = ("word", "char", "combined")
LOCK_FILES = ("triage.py", "evaluate.py", "experiment.py", "results/policy.json")


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def class_scores(rows, categories):
    pairs = [(row, answer) for row, answer in zip(rows, categories) if row["label"] != REVIEW]
    true, pred = zip(*pairs)
    return classify_metrics(true, pred, list(CATEGORIES))


def evaluate_model(model, policy, rows):
    outcomes = []
    for row in rows:
        p = predict(model, row["text"])
        queue, reason = route(p, policy)
        outcomes.append({
            "predicted": p.category, "score": round(p.score, 6), "second": p.second_category,
            "gap": round(p.gap, 6), "route": queue, "reason": reason,
        })
    categories = [item["predicted"] for item in outcomes]
    queues = [item["route"] for item in outcomes]
    return {"classification": class_scores(rows, categories),
            "routing": route_metrics(rows, queues)}, outcomes


def lock(train, validation):
    settings = {}
    for variant in VARIANTS:
        model = fit_model(train, variant)
        policy = load_policy() if variant == "combined" else fit_policy(model, validation)
        measured, _ = evaluate_model(model, policy, validation)
        settings[variant] = {"policy": policy, "validation": measured}
        print(f"{variant:8} validation F1={measured['classification']['macro_f1']:.3f} "
              f"auto={measured['routing']['auto_correct']}/{measured['routing']['auto_count']}")
    save_json("experiment_protocol.json", {
        "method": "Варианты зафиксированы на train/validation. combined использует прежние пороги.",
        "train_sha256": sha_file("train"), "validation_sha256": sha_file("validation"),
        "files": {p: digest(p) for p in LOCK_FILES}, "variants": settings,
    })


def stress_breakdown(rows, base, decisions):
    groups = {}
    for row, keyword, result in zip(rows, base, decisions):
        group = groups.setdefault(row["phenomenon"], Counter())
        group["n"] += 1
        group["review_expected"] += row["label"] == REVIEW
        group["keywords_correct"] += keyword == row["label"]
        group["ml_correct"] += result["route"] == row["label"]
        group["ml_manual"] += result["route"] == REVIEW
        group["ml_wrong_auto"] += result["route"] != REVIEW and result["route"] != row["label"]
    return groups


def timing(model, rows):
    durations = []
    for _ in range(5):
        for row in rows:
            start = time.perf_counter_ns()
            predict(model, row["text"])
            durations.append((time.perf_counter_ns() - start) / 1_000_000)
    return {"calls": len(durations), "median_ms": statistics.median(durations),
            "p95_ms": statistics.quantiles(durations, n=100)[94]}


def run_evaluation(train, validation):
    import json
    protocol = json.loads((ROOT / "results" / "experiment_protocol.json").read_text(encoding="utf-8"))
    for split in ("train", "validation"):
        if sha_file(split) != protocol[f"{split}_sha256"]:
            raise ValueError(f"Изменились данные {split}. Нужен новый протокол.")
    if any(digest(path) != signature for path, signature in protocol["files"].items()):
        raise ValueError("Изменился исходный код или пороги после фиксации протокола.")
    samples = {name: read_split(name) for name in ("test", "stress")}
    check_data({"train": train, "validation": validation, **samples})
    for row in samples["stress"]:
        assert row["phenomenon"] and row["expected_reason"], row["id"]

    models = {variant: fit_model(train, variant) for variant in VARIANTS}
    summary = {"environment": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                               "platform": platform.system()}, "splits": {}}
    for split, rows in samples.items():
        keywords = [baseline(row["text"]) for row in rows]
        metrics = {"keywords": {"classification": class_scores(rows, keywords),
                               "routing": route_metrics(rows, keywords)}}
        outcomes = {}
        for variant, model in models.items():
            policy = protocol["variants"][variant]["policy"]
            metrics[variant], outcomes[variant] = evaluate_model(model, policy, rows)
        predictions = []
        for index, row in enumerate(rows):
            prediction = {**row, "keywords": keywords[index]}
            for variant in VARIANTS:
                prediction.update({f"{variant}_{field}": value
                                   for field, value in outcomes[variant][index].items()})
            predictions.append(prediction)
        with (ROOT / "results" / f"extended_{split}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        summary["splits"][split] = {"sha256": sha_file(split), "n": len(rows), "metrics": metrics}
        if split == "stress":
            summary["stress_groups"] = stress_breakdown(rows, keywords, outcomes["combined"])
            summary["timing_on_stress"] = timing(models["combined"], rows)
        print(f"=== {split}, n={len(rows)} ===")
        for method, result in metrics.items():
            a, b = result["classification"], result["routing"]
            print(f"{method:8} topics={a['correct']}/{a['total']} f1={a['macro_f1']:.3f}  "
                  f"queues={b['correct']}/{b['total']}  auto={b['auto_correct']}/{b['auto_count']}  "
                  f"review={b['review_held']}/{b['review_expected']}")
    save_json("extended_metrics.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", action="store_true", help="Зафиксировать только train/validation")
    args = parser.parse_args()
    train, validation = read_split("train"), read_split("validation")
    check_data({"train": train, "validation": validation})
    if args.lock:
        lock(train, validation)
    else:
        run_evaluation(train, validation)

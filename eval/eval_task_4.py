#!/usr/bin/env python
# coding: utf-8

import json
import os
import warnings
from sys import argv

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score.rouge_scorer import RougeScorer

warnings.filterwarnings("ignore")


def recursive_path_finder(start_path: str, directory_name: str) -> list[str]:
    def _recursive_path_finder(start_path_: str) -> None:
        for path_entry in os.listdir(start_path_):
            path = os.path.join(start_path_, path_entry)

            if os.path.isdir(path):
                if directory_name in os.listdir(path) or path_entry != directory_name:
                    # If the directory name is in the list of entries, or the entry is not the directory name
                    _recursive_path_finder(path)
                else:
                    # If the entry is the directory name
                    path_entries.append(path)

    path_entries = []
    _recursive_path_finder(start_path)

    return path_entries


def calculate_bleu(prediction: str, ground_truth: str) -> float:
    prediction_tokens = prediction.lower().strip().split()
    ground_truth_tokens = ground_truth.lower().strip().split()

    return sentence_bleu([ground_truth_tokens], prediction_tokens)


def calculate_rouge(scorer: RougeScorer, prediction: str, ground_truth: str) -> tuple[float, float, float]:
    prediction = prediction.lower().strip()
    ground_truth = ground_truth.lower().strip()

    result = scorer.score(prediction, ground_truth)

    return result["rouge1"].fmeasure, result["rouge2"].fmeasure, result["rougeL"].fmeasure


def evaluate(task_path_entries: list[str], output_path: str) -> None:
    all_result = {}
    tasks = ("4-1", "4-2", "4-3", "4-4")
    tasks_type = ("language", "rating", "language", "rating")
    keys = ["bleu", "rouge-1", "rouge-2", "rouge-l", "rmse", "mae"]

    for task_path_entry in task_path_entries:
        print(f"Processing {task_path_entry}")

        result = {"bleu": [], "rouge-1": [], "rouge-2": [], "rouge-l": [], "rmse": [], "mae": []}

        for task, task_type, path_entry in zip(tasks, tasks_type, sorted(
                [x for x in os.listdir(task_path_entry) if os.path.isdir(os.path.join(task_path_entry, x))],
                key=lambda x: int(x.split("-")[-1])
        )):
            print(f" Processing {path_entry}")

            output = json.load(open(os.path.join(task_path_entry, path_entry, "results.json"), "r", encoding="utf-8"))

            prompts = []
            pred_explanations = []
            pred_ratings = []
            gt_explanations = []
            gt_ratings = []

            for entry in output:
                prompts.append(entry["source_text"])
                prediction = entry["pred"]
                ground_truth = entry["gt"]

                if task_type == "language":
                    pred_explanations.append(prediction)
                    gt_explanations.append(ground_truth)
                else:
                    try:
                        pred_rating = float(prediction)
                        gt_rating = float(ground_truth)
                    except ValueError:
                        pred_ratings.append(None)
                        gt_ratings.append(None)
                    else:
                        pred_ratings.append(pred_rating)
                        gt_ratings.append(gt_rating)

            if task_type == "language":
                scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

                bleu = tuple(
                        calculate_bleu(prediction, ground_truth) for prediction, ground_truth in
                        zip(pred_explanations, gt_explanations)
                )
                rouge = tuple(
                        calculate_rouge(scorer, prediction, ground_truth) for prediction, ground_truth in
                        zip(pred_explanations, gt_explanations)
                )
                rouge_1, rouge_2, rouge_l = zip(*rouge)

                print(f"  BLEU: {sum(bleu) / len(bleu)}")
                print(f"  ROUGE-1: {sum(rouge_1) / len(rouge_1)}")
                print(f"  ROUGE-2: {sum(rouge_2) / len(rouge_2)}")
                print(f"  ROUGE-L: {sum(rouge_l) / len(rouge_l)}")

                result["bleu"].append(sum(bleu) / len(bleu))
                result["rouge-1"].append(sum(rouge_1) / len(rouge_1))
                result["rouge-2"].append(sum(rouge_2) / len(rouge_2))
                result["rouge-l"].append(sum(rouge_l) / len(rouge_l))
                result["rmse"].append(None)
                result["mae"].append(None)

            elif task_type == "rating":
                invalid_ratings = 0
                rmse = []
                mae = []

                for prediction, ground_truth in zip(pred_ratings, gt_ratings):
                    if prediction is None or ground_truth is None:
                        invalid_ratings += 1
                    else:
                        rmse.append((prediction - ground_truth) ** 2)
                        mae.append(abs(prediction - ground_truth))

                print(f"  RMSE: {sum(rmse) / len(rmse)}")
                print(f"  MAE: {sum(mae) / len(mae)}")
                print(f"  Invalid ratings: {invalid_ratings} ({invalid_ratings / len(pred_ratings) * 100:.2f}%)")

                result["rmse"].append(sum(rmse) / len(rmse))
                result["mae"].append(sum(mae) / len(mae))
                result["bleu"].append(None)
                result["rouge-1"].append(None)
                result["rouge-2"].append(None)
                result["rouge-l"].append(None)

        result = pd.DataFrame(result, index=list(tasks))
        result["bleu"] *= 100
        result["rouge-1"] *= 100
        result["rouge-2"] *= 100
        result["rouge-l"] *= 100

        # Add a row for the average
        result.loc["average"] = result.mean()
        experiment_name = "/".join(task_path_entry.split(os.sep)[-5:])
        all_result[experiment_name] = result

        result.to_csv(os.path.join(task_path_entry, "metrics.csv"), index=True, index_label="task")

    columns = ["epoch", "token_method", "dataset", "trained_task", "prompt_id"]
    columns.extend(keys)
    final_result = pd.DataFrame(columns=columns)
    for experiment_path in all_result:
        epoch, token_method, dataset, trained_task, _ = experiment_path.split("/")

        # Concatenate the results
        results = all_result[experiment_path]
        results["epoch"] = epoch
        results["token_method"] = token_method
        results["dataset"] = dataset
        results["trained_task"] = trained_task
        results["prompt_id"] = results.index
        final_result = pd.concat([final_result, results], axis=0)

    final_result.to_csv(os.path.join(output_path, "metrics_task4.csv"), index=False)
    # json.dump(all_result, open(os.path.join(output_path, "metrics_task3.json"), "w", encoding="utf-8"), indent=4)


def check_missing_files(task_path_entries: list[str]) -> None:
    # Check if "results.json" exists in each task folder
    missing_files_path = []

    for task_path_entry in task_path_entries:
        for path_entry in os.listdir(task_path_entry):
            if not os.path.isdir(os.path.join(task_path_entry, path_entry)):
                continue

            for task_entry in os.listdir(os.path.join(task_path_entry, path_entry)):
                if not os.path.isdir(os.path.join(task_path_entry, path_entry, task_entry)):
                    continue

                if not os.path.isfile(os.path.join(task_path_entry, path_entry, task_entry, "results.json")):
                    missing_files_path.append(os.path.join(task_path_entry, path_entry, task_entry))

    if len(missing_files_path) > 0:
        err_msg = "The following folders are missing the \"results.json\" file:\n"
        for path in missing_files_path:
            err_msg += f" - {path}\n"

        raise FileNotFoundError(err_msg)


def main():
    root_path = argv[1]
    output_path = argv[2]

    task_path_entries = recursive_path_finder(root_path, "task-4")
    check_missing_files(task_path_entries)

    print("Evaluating the following experiments:")
    for task_path_entry in task_path_entries:
        print(f" - {task_path_entry}")

    evaluate(task_path_entries, output_path)


if __name__ == '__main__':
    main()

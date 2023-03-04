import argparse
import json
import warnings
import os

import pandas as pd

from eval_utils import evaluate_binary, evaluate_rating, recursive_path_finder

warnings.filterwarnings("ignore")


def main(path: str):

    # configure
    task = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10']
    binary_prompt = ['1-3', '1-4', '1-8', '1-9']
    all_result = {"rmse": [], "mae": [], "acc": [], "f1": [], "precision": [], "recall": [], "invalid": [],
                  "rating_1": [], "rating_2": [], "rating_3": [], "rating_4": [], "rating_5": []}

    output_path = os.path.join(
            os.path.pardir,
            path
    )
    for task_type in task:
        print(f"Processing {task_type}")

        # load data from output file
        output = json.load(open(os.path.join(path, task_type, "results.json"), "r", encoding="utf-8"))
        prompt, pred, gt = [], [], []
        for entry in output:
            prompt.append(entry["source_text"])
            pred.append(entry["pred"])
            gt.append(entry["gt"])

        # evaluate
        if task_type in binary_prompt:
            evaluation = evaluate_binary(pred, gt)

            for key, value in evaluation.items():
                print(f"  {key}: {value}")

            for metric in all_result.keys():
                if metric in evaluation:
                    all_result[metric].append(evaluation[metric])
                else:
                    all_result[metric].append(None)

        else:
            evaluation = evaluate_rating(pred, gt)

            for key, value in evaluation.items():
                print(f"  {key}: {value}")

            for metric in all_result.keys():
                if metric in evaluation:
                    all_result[metric].append(evaluation[metric])
                else:
                    all_result[metric].append(None)

    # save to csv
    all_result = pd.DataFrame(all_result, index=task)
    all_result.loc['average'] = all_result.mean()
    all_result.to_csv(os.path.join(output_path, "metric.csv"), index=True, index_label="task")

    return all_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="output path")
    args = parser.parse_args()

    task_path_entries = recursive_path_finder(args.path, "task-1")

    all_results = {"/".join(task_path_entry.split(os.sep)[-5:]): main(task_path_entry) for task_path_entry in
                   task_path_entries}

    # Write the results to a JSON file
    for experiment_path in all_results:
        all_results[experiment_path] = all_results[experiment_path]

    columns = ["epoch", "token_method", "dataset", "trained_task", "prompt_id", "rmse", "mae", "acc", "f1", "precision", "recall",
               "invalid", "rating_1", "rating_2", "rating_3", "rating_4", "rating_5"]
    final_results = pd.DataFrame(columns=columns)
    for experiment_path in all_results:
        epoch, token_method, dataset, trained_task, _ = experiment_path.split("/")

        # Concatenate the results
        results = all_results[experiment_path]
        results["epoch"] = epoch
        results["token_method"] = token_method
        results["dataset"] = dataset
        results["trained_task"] = trained_task
        results["prompt_id"] = results.index
        final_results = pd.concat([final_results, results], axis=0)

    final_results.to_csv(os.path.join("metrics_task1.csv"), index=False)
    # final_results.to_excel(os.path.join("metrics_task1.xlsx"), index=False)

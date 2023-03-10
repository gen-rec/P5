import argparse
import json
import os
import warnings
from multiprocessing import Pool

import pandas as pd

from eval_utils import evaluate_binary, evaluate_rating, evaluate_sequential, recursive_path_finder

warnings.filterwarnings("ignore")

keys = ["hit@1", "hit@5", "ndcg@5", "hit@10", "ndcg@10", "acc", "f1", "precision", "recall", "invalid"]


def main(path: str):

    # configure for direct recommendation
    task = ["5-1", "5-2", "5-3", "5-4", "5-5", "5-6", "5-7", "5-8"]
    binary_prompt = ["5-1", "5-2", "5-3", "5-4"]
    all_result = {"hit@1": [], "hit@5": [], "ndcg@5": [], "hit@10": [], "ndcg@10": [], "acc": [], "f1": [],
                  "precision": [], "recall": [], "invalid": []}

    output_path = os.path.join(
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
            evaluation = evaluate_sequential(pred, gt, k_list=[1, 5, 10])

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

    task_path_entries = recursive_path_finder(args.path, "task-5")

    all_keys = ["/".join(task_path_entry.split(os.sep)[-5:]) for task_path_entry in task_path_entries]
    with Pool(8) as p:
        all_results = dict(zip(all_keys, p.map(main, task_path_entries)))

    # Write the results to a JSON file
    for experiment_path in all_results:
        all_results[experiment_path] = all_results[experiment_path]

    columns = ["epoch", "token_method", "dataset", "trained_task", "prompt_id"]
    columns.extend(keys)
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

    final_results.to_csv(os.path.join("metrics_task5.csv"), index=False)

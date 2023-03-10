import argparse
import json
import os
import warnings

import pandas as pd

from multiprocessing import Pool

from eval_utils import evaluate_binary, evaluate_rating, evaluate_sequential, recursive_path_finder, evaluate_generation

warnings.filterwarnings("ignore")

keys = ["bleu-4", "rouge-1", "rouge-2", "rouge-l", "rmse", "mae", "invalid"]
keys += [f"rating_{i}" for i in range(1, 6)]


def main(path: str):

    # configure for explanation generation
    tasks = [f"3-{i}" for i in range(1, 13)]
    with_rating_prediction = ["3-7", "3-8"]
    all_result = {k: [] for k in keys}

    output_path = os.path.join(
            path
    )
    for task_type in tasks:
        print(f"Processing {task_type}")

        # load data from output file
        output = json.load(open(os.path.join(path, task_type, "results.json"), "r", encoding="utf-8"))
        prompt, pred, gt = [], [], []
        for entry in output:
            prompt.append(entry["source_text"])
            pred.append(entry["pred"])
            gt.append(entry["gt"])

        # evaluate
        if task_type in with_rating_prediction:
            # Split predicted rating and generated explanation
            pred_rp = []  # Predicted rating
            pred_gen = []  # Predicted explanation
            gt_rp = []  # Ground truth rating
            gt_gen = []  # Ground truth explanation
            invalid = 0  # Number of entries where the predicted rating is not a number

            for p, g in zip(pred, gt):
                try:
                    p_rating = float(p.split(",", maxsplit=1)[0])
                    g_rating = float(g.split(", ", maxsplit=1)[0])

                except ValueError:
                    invalid += 1
                    pred_gen.append(p)
                    gt_gen.append(g.split(", ", maxsplit=1)[1])

                else:
                    try:
                        _, pred_gen_ = p.split(",", maxsplit=1)
                        _, gt_gen_ = g.split(", ", maxsplit=1)
                    except IndexError or ValueError:
                        pass
                    else:
                        pred_rp.append(p_rating)
                        gt_rp.append(g_rating)
                        pred_gen.append(pred_gen_)
                        gt_gen.append(gt_gen_)

            evaluation_rating = evaluate_rating(pred_rp, gt_rp)
            evaluation_rating["invalid"] = invalid

            pred = pred_gen
            gt = gt_gen
        else:
            evaluation_rating = {}

        evaluation = evaluate_generation(pred, gt)
        evaluation.update(evaluation_rating)

        for key, value in evaluation.items():
            print(f"  {key}: {value}")

        for metric in all_result.keys():
            if metric in evaluation:
                all_result[metric].append(evaluation[metric])
            else:
                all_result[metric].append(None)

    # save to csv
    all_result = pd.DataFrame(all_result, index=tasks)
    all_result.loc['average'] = all_result.mean()
    all_result.to_csv(os.path.join(output_path, "metric.csv"), index=True, index_label="task")

    return all_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="output path")
    args = parser.parse_args()

    task_path_entries = recursive_path_finder(args.path, "task-3")

    # all_results = {"/".join(task_path_entry.split(os.sep)[-5:]): main(task_path_entry) for task_path_entry in
    #                task_path_entries}
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

    final_results.to_csv(os.path.join("metrics_task3.csv"), index=False)
    # final_results.to_excel(os.path.join("metrics_task1.xlsx"), index=False)

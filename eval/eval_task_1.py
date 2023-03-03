
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from eval.utils import evaluate_rating, evaluate_binary
import argparse
import json


def main(path:str):

    # configure
    task = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10']
    binary_prompt = ['1-3', '1-4', '1-8', '1-9']
    all_result = {"rmse": [], "mae": [], "acc" : [], "acc": [], "f1": [], "precision": [], "recall": [], "invalid": []}


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

            print(f"  Accuracy: {evaluation['acc']}")
            print(f"  F1: {evaluation['f1']}")
            print(f"  Precision: {evaluation['precision']}")
            print(f"  Recall: {evaluation['recall']}")
            print(f"  Invalid output: {evaluation['invalid']} ({evaluation['invalid'] / len(pred) * 100:.2f}%)")

            for metric in all_result.keys():
                if metric in evaluation:
                    all_result[metric].append(evaluation[metric])
                else:
                    all_result[metric].append(None)
            
        else:
            evaluation = evaluate_rating(pred, gt)

            print(f"  RMSE: {evaluation['rmse']}")
            print(f"  MAE: {evaluation['mae']}")
            print(f"  Invalid output: {evaluation['invalid']} ({evaluation['invalid'] / len(pred) * 100:.2f}%)")

            for metric in all_result.keys():
                if metric in evaluation:
                    all_result[metric].append(evaluation[metric])
                else:
                    all_result[metric].append(None)

        # save to csv
        all_result = pd.DataFrame(all_result, index=task)
        all_result.loc['average'] = all_result.mean()
        all_result.to_csv(os.path.join(output_path, "metric.csv"), index=True, index_label="task")
        

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="output path")
    args = parser.parse_args()

    main(path=args.path)
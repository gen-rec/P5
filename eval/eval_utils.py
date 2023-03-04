import os

from metrics.utils import bleu_score, rouge_score
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def evaluate_rating(pred, gt, min_rating=1, max_rating=5):
    rmse, mae = [], []
    ratings = {i: 0 for i in range(1, 6)}
    invalid = 0
    for p, g in zip(pred, gt):
        try:
            if float(p) <= min_rating or float(p) >= max_rating:
                invalid += 1
                continue
            rmse.append((float(p) - float(g)) ** 2)
            mae.append(abs(float(p) - float(g)))

            ratings[int(round(float(p)))] += 1

        except ValueError:
            invalid += 1

    rmse, mae = sum(rmse) / len(rmse), sum(mae) / len(mae)

    result = {f"rating_{i}": v for i, v in enumerate(ratings.values(), start=1)}
    result.update({
        "rmse": rmse,
        "mae": mae,
        "invalid": invalid,
    })

    return result


def evaluate_sequential(pred, gt, k_list=None):
    if k_list is None:
        k_list = [5, 10]

    result = {}

    for k in k_list:
        hit, ndcg = [], []
        for p, g in zip(pred, gt):
            hit.append(1 if g in p[:k] else 0)
            ndcg.append(1 / (p[:k].index(g) + 1) if g in p[:k] else 0)

        hit, ndcg = sum(hit) / len(hit), sum(ndcg) / len(ndcg)

        result[f"hit@{k}"] = hit
        result[f"ndcg@{k}"] = ndcg

    return result


def evaluate_generation(pred: list[str], gt: list[str]):
    pred = [p.strip() for p in pred]
    gt = [g.strip() for g in gt]
    rouge = rouge_score(pred, gt)

    pred = [p.split() for p in pred]
    gt = [g.split() for g in gt]
    bleu_4 = bleu_score(pred, gt, n_gram=4)

    rouge_1 = rouge["rouge_1/f_score"]
    rouge_2 = rouge["rouge_2/f_score"]
    rouge_l = rouge["rouge_l/f_score"]

    return {
        "bleu-4": bleu_4,
        "rouge-1": rouge_1,
        "rouge-2": rouge_2,
        "rouge-l": rouge_l,
    }


def evaluate_binary(pred, gt):
    valid_pred, valid_gt = [], []
    invalid = 0
    for p, g in zip(pred, gt):
        if p in ['yes', 'no'] and g in ['yes', 'no']:
            valid_pred.append(p)
            valid_gt.append(g)
        elif p in ['like', 'dislike'] and g in ['like', 'dislike']:
            valid_pred.append(p)
            valid_gt.append(g)
        else:
            invalid += 1

    acc = accuracy_score(valid_gt, valid_pred)
    f1 = f1_score(valid_gt, valid_pred, average='macro')
    precision = precision_score(valid_gt, valid_pred, average='macro')
    recall = recall_score(valid_gt, valid_pred, average='macro')

    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "invalid": invalid,
    }


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

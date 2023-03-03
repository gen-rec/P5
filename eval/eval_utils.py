
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score

def evaluate_rating(pred, gt, min=1, max=5):
    rmse, mae = [], []
    invalid = 0
    for p, g in zip(pred, gt):
        try:
            if float(p) <= min or float(p) >= max:
                invalid += 1
                continue
            rmse.append((float(p) - float(g)) ** 2)
            mae.append(abs(float(p) - float(g)))
        except ValueError:
            invalid += 1

    rmse, mae = sum(rmse) / len(rmse), sum(mae) / len(mae)

    return {
        "rmse": rmse,
        "mae": mae,
        "invalid": invalid,
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

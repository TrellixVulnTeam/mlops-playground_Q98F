import argparse
import json
import pickle
import tarfile
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


def get_prediction(score, threshold=0.5):
    return np.where(score >= threshold, 1, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/opt/ml/processing", type=str)

    args = parser.parse_args()
    base_dir = args.base_dir

    model_path = f"{base_dir}/models"
    with tarfile.open(model_path + "/model.tar.gz") as tar:
        tar.extractall(path=model_path)

    estimator = pickle.load(open(model_path + "/xgboost-model", "rb"))
    arr_test = np.loadtxt(f"{base_dir}/test/arr_test.csv", delimiter=",")

    y_test, X_test = arr_test[:, 0], arr_test[:, 1:]
    X_test = xgb.DMatrix(X_test)
    scores = estimator.predict(X_test)
    predictions = get_prediction(scores)

    eval_metrics = {
        "eval_metric": {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "auroc": roc_auc_score(y_test, scores),
            "auprc": average_precision_score(y_test, scores),
        }
    }

    with open(f"{base_dir}/eval/eval_metrics.json", "w") as f:
        f.write(json.dumps(eval_metrics))

import argparse
import json

# import pickle
import tarfile
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    f1_score,
    roc_auc_score,
)

np.random.seed(42)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def get_prediction(score, threshold=0.5):
    return np.where(score >= threshold, 1, 0)


def get_sampled_curve(curve, sampling_rate=0.1, n_digit=6):
    length = curve[0].shape[0]
    indices = np.sort(
        np.random.choice(
            np.arange(length),
            int(length * sampling_rate),
            replace=False,
        )
    )
    return np.round(curve[0], n_digit)[indices], np.round(curve[1], n_digit)[indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/opt/ml/processing", type=str)

    args = parser.parse_args()
    base_dir = args.base_dir

    model_path = f"{base_dir}/models"
    with tarfile.open(model_path + "/model.tar.gz") as tar:
        tar.extractall(path=model_path)

    # The process of saving and loading models has changed since XGBoost version 1.3.x.
    estimator = xgb.Booster()
    estimator.load_model(model_path + "/xgboost-model")
    arr_test = np.loadtxt(f"{base_dir}/test/arr_test.csv", delimiter=",")
    # estimator = pickle.load(open(model_path + "/xgboost-model", "rb"))

    y_test, X_test = arr_test[:, 0], arr_test[:, 1:]
    X_test = xgb.DMatrix(X_test)
    scores = estimator.predict(X_test)
    predictions = get_prediction(scores)

    conf_mat = confusion_matrix(y_test, predictions)
    roc_curv = get_sampled_curve(roc_curve(y_test, scores))
    pr_curv = get_sampled_curve(precision_recall_curve(y_test, scores))

    eval_metrics = {
        "binary_classification_metrics": {
            "confusion_matrix": {
                "0": {"0": conf_mat[0][0], "1": conf_mat[0][1]},
                "1": {"0": conf_mat[1][0], "1": conf_mat[1][1]},
            },
            "recall": {
                "value": recall_score(y_test, predictions),
                "standard_deviation": "NaN",
            },
            "precision": {
                "value": precision_score(y_test, predictions),
                "standard_deviation": "NaN",
            },
            "accuracy": {
                "value": accuracy_score(y_test, predictions),
                "standard_deviation": "NaN",
            },
            "true_positive_rate": {
                "value": conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1]),
                "standard_deviation": "NaN",
            },
            "true_negative_rate": {
                "value": conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1]),
                "standard_deviation": "NaN",
            },
            "false_positive_rate": {
                "value": conf_mat[0][1] / (conf_mat[0][0] + conf_mat[0][1]),
                "standard_deviation": "NaN",
            },
            "false_negative_rate": {
                "value": conf_mat[1][0] / (conf_mat[1][0] + conf_mat[1][1]),
                "standard_deviation": "NaN",
            },
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": roc_curv[0],
                "true_positive_rates": roc_curv[1],
            },
            "precision_recall_curve": {
                "precisions": pr_curv[0],
                "recalls": pr_curv[1],
            },
            "auc": {
                "value": roc_auc_score(y_test, scores),
                "standard_deviation": "NaN",
            },
            "f1": {
                "value": f1_score(y_test, predictions),
                "standard_deviation": "NaN",
            },
        }
    }

    with open(f"{base_dir}/eval/eval_metrics.json", "w") as f:
        f.write(json.dumps(eval_metrics, cls=NumpyEncoder))

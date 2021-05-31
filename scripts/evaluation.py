
import json
import pathlib
import pickle
import tarfile
import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             average_precision_score, roc_auc_score)


def get_prediction(score, thr=0.5):
    return np.where(score >= thr, 1, 0)


if __name__ == '__main__':
    base_dir = '/opt/ml/processing'
    
    model_path = f'{base_dir}/models'
    with tarfile.open(model_path + '/model.tar.gz') as tar:
        tar.extractall(path=model_path)
        
    estimator = pickle.load(open(model_path + '/xgboost-model', 'rb'))
    
    data_path = f'{base_dir}/test/dtest.csv'
    dtest = np.loadtxt(data_path, delimiter=',')
    
    y_test, X_test = dtest[:, 0], dtest[:, 1:]
    X_test = xgb.DMatrix(X_test)
    scores = estimator.predict(X_test)
    predictions = get_prediction(scores)
 
    eval_metrics = {
        'eval_metric': {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions),
            'auroc': roc_auc_score(y_test, scores),
            'auprc': average_precision_score(y_test, scores)
        }
    }
    
    output_dir = f'{base_dir}/eval'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_path = f'{output_dir}/eval_metrics.json'
    with open(output_path, 'w') as f:
        f.write(json.dumps(eval_metrics))

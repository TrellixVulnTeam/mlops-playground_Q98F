
import json
import pathlib
import pickle
import tarfile
import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             average_precision_score, roc_auc_score)


def get_pred(score, thr=0.5):
    return np.where(score >= thr, 1, 0)


if __name__ == '__main__':
    base_dir = '/opt/ml/processing'
    
    model_path = f'{base_dir}/models/model.tar.gz'
    with tarfile.open(model_path) as tar:
        tar.extractall(path='.')
        
    clf = pickle.load(open('xgboost-model', 'rb'))
    
    data_path = f'{base_dir}/test/dtest.csv'
    dtest = np.loadtxt(data_path, delimiter=',')
    
    y_test, X_test = dtest[:, 0], dtest[:, 1:]
    X_test = xgb.DMatrix(X_test)
    scores = clf.predict(X_test)
    preds = get_pred(scores)
 
    eval_metrics = {
        'eval_metric': {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds),
            'f1': f1_score(y_test, preds),
            'auroc': roc_auc_score(y_test, scores),
            'auprc': average_precision_score(y_test, scores)
        }
    }
    
    output_dir = f'{base_dir}/eval'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_path = f'{output_dir}/eval_metrics.json'
    with open(output_path, 'w') as f:
        f.write(json.dumps(eval_metrics))

import json
import pathlib
import pickle
import tarfile
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             average_precision_score, roc_auc_score)


def get_pred(score, thr=0.5):
    return np.where(score >= thr, 1, 0)


if __name__ == '__main__':
    model_path = f'/opt/ml/processing/models/model.tar.gz'
    with tarfile.open(model_path) as tar:
        tar.extractall(path='.')
        
    clf = pickle.load(open('xgboost-model', 'rb'))
    
    data_path = '/opt/ml/processing/test/dtest.csv'
    np.loadtxt(data_path, delimiter=',')
    
    dtest = pd.read_csv(data_path, header=None)
    y_test, X_test = dtest[:, 0], dtest[:, 1:]
    
    scores = clf.predict(xgb.DMatrix(X_test))
    preds = get_pred(scores)
    
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auroc = roc_auc_score(y_test, scores)
    auprc = average_precision_score(y_test, scores)

    report_dict = {
        'eval_metric': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auroc': auroc,
            'auprc': auprc
        }
    }
    
    output_dir = '/opt/ml/processing/evaluation'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_path = f'{output_dir}/evaluation.json'
    with open(output_path, 'w') as f:
        f.write(json.dumps(report_dict))

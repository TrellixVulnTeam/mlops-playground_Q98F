# IEEE-CIS Fraud Detection with Sagemaker Pipelines
#### Goals
* Perform common predictive modeling with a SageMaker *Estimator* and *HyperparameterTuner*.
* Use *SageMaker Pipelines* to orchestrate predictive modeling workflows.
  
#### Requirements
* The dataset can be downloaded from [this Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection).
* In addition to the [Anaconda](https://www.anaconda.com) libraries, you need to install `category_encoders`, `selenium`, `geckodriver` and `scikit-learn` version 0.24 or higher.
* You also need to set up an AWS account and install `awscli` and `sagemaker-python-sdk`.

## SageMaker Estimator and HyperparameterTuner
### Data Splitting and Preprocessing
I preprocessed categorical variables with missing value imputation and ordinal encoding. In the case of *XGBoost*, there is no logic to separately process categorical variables, so I quantified them with the target encoding of `category_encoders`. I simply preprocessed numerical variables with missing value imputation.
  
### Hyperparameter Tuning
First, I defined an *Estimator* with SageMaker's *XGBoost framework*. Then, I efficiently obtained the optimal hyperparameter values through *HyperparameterTuner*. Below are the convergence plot and the exploration results for each parameter.
* Reference: [Analyze Results of a Hyperparameter Tuning Job](https://github.com/aws/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb)

![Convergence Plot](img/convergence_plot.svg)

![Partial Dependence Plot](img/partial_dependence_plot.png)

### Model Evaluation
After defining the *Transformer* of the optimal *Estimator*, I obtained the predicted values in batches through *Batch Transform*. By comparing them with labels, prediction performance can be obtained, which are the confusion matrix, ROC curve, and Precision - Recall curve drawn below.

![Confusion Matrix](img/conf_mat.svg)

![ROC Curve Plot](img/roc_curve.svg)

![PR Curve Plot](img/pr_curve.svg)

### Model Re-training
Finally, after retraining the model by combining the training set and the test set, I predicted the unlabeled dataset and submitted the scores.

## SageMaker Pipelines

![Pipeline DAG](img/pipeline_dag.png)

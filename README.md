# IEEE-CIS Fraud Detection with Sagemaker Pipelines
### Goals
* Perform common predictive modeling with a SageMaker **Estimator**, **HyperparameterTuner**, and **Clarify**.
* Use **SageMaker Pipelines** to orchestrate predictive modeling workflow.
  
### Requirements
* The dataset can be downloaded from [this Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection).
* In addition to the [Anaconda](https://www.anaconda.com) libraries, you need to install `category_encoders`, `selenium`, `geckodriver` and `scikit-learn` version 0.24 or higher.
* You also need to set up an AWS account and install `awscli`, `sagemaker-python-sdk`, and `sagemaker-inference-toolkit`.

## SageMaker Estimator, HyperparameterTuner, and Clarify
### Data Splitting and Preprocessing
The features have been properly preprocessed. In particular, since *[XGBoost](https://arxiv.org/pdf/1603.02754.pdf)* does not have the ability to directly process categorical features, they were quantified using *[Target Encoding](https://dl.acm.org/doi/10.1145/507533.507538)* of `category_encoders`.
  
### Hyperparameter Tuning
First, **Estimator** was defined as SageMaker's **XGBoost framework**. The optimal hyperparameter values were then efficiently obtained by fitting a **HyperparameterTuner** that automatically initiates a *Bayesian search*. Here is the convergence plot and the search results for each parameter.

![Convergence Plot](img/convergence_plot.svg)

![Partial Dependence Plot](img/partial_dependence_plot.png)

### Model Evaluation
After defining the **Transformer** of the optimal **Estimator**, the predicted values were obtained in batches through **Batch Transform**. By comparing them to labels, prediction performance can be measured, which is the confusion matrix, ROC curve, and Precision - Recall curve plotted below.
  
| Accuracy | Precision | Recall |   F1   | AUROC  | AUPRC  |    
|:--------:|:---------:|:------:|:------:|:------:|:------:|
|  98.86%  |  93.80%   | 72.25% | 81.63% | 97.38% | 86.93% |  

![Confusion Matrix](img/conf_mat.svg)

![ROC Curve Plot](img/roc_curve.svg)

![PR Curve Plot](img/pr_curve.svg)

### Model Retraining
Finally, after retraining the model by combining the training and test sets, unlabeled dataset was predicted and scores were submitted to Kaggle. (AUROC 90.63% has been achieved.)

### Model Explanation
*[Shapley values](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)* were calculated using **Clarify** for model explainability. As a baseline, the averages of the training set features was used, and due to computational complexity, they were calculated by sampling only a subset of the dataset. Below are the top 25 features with the highest global Shapley values.
![Global Shapley Values](img/global_shap_values.svg)

## SageMaker Pipelines
* I created a custom image to use `scikit-learn` version 0.24 and `category_encoders` library. To build the image with the pre-made `Dockerfile` and push it to Amazon ECR, you need to run the shell script named `run.sh`. As a result, instead of **SKLearnProcessor** with framework version 0.23, you can run a custom image-based **ScriptProcessor** with the required libraries.
* This is a special case where the model is trained every time a batch transform is performed.
* a) You can run a **TrainStep** to fit the **Estimator** with default hyperparameter values. In this case, the model is refitted by merging up to the test set, and prediction scores are calculated. b) Otherwise, it is possible to execute a **TunerStep** to fit the **HyperparameterTuner**. In this case, prediction scores are calculated using the optimal model obtained in the tuning process without refitting.  
    
|Name|Step|Base Job Class|Description|
|:---:|:---:|:---:|:---:|
|PreprocessData|**ProcessingStep**|~~SKLearnProcessor~~ **ScriptProcessor**|Data splitting and preprocessing|
|TrainModel|**TrainingStep**|**Estimator**|A *XGBoost* **Estimator** fitting|
|TuneHyperparameters|**TunerStep**|**HyperparameterTuner**|Hyperparameter tuning|
|EvaluateModel|**ProcessingStep**|**ScriptProcessor**|The fitted **Estimator** evaluation saved in a *JSON* **PropertyFile**|
|CheckCondition|**ConditionStep**| |A target metric checking to conditionally perform subsequent steps|
|Re-preprocessData|**ProcessingStep**|~~SKLearnProcessor~~ **ScriptProcessor**|Data repreprocessing|
|Re-trainModel|**TrainingStep**|**Estimator**|A *XGBoost* **Estimator** refitting|
|RegisterModel|**RegisterModel**| |Model packing and registration in a *ModelPackageGroup* with **ModelMetrics**|
|DeployModel|**CreateModel**| |Model deployment|
|PredictData|**TransformStep**|**Transformer**|Batch transformation|

![Pipeline DAG](img/dag.png)

* References:   
  [XGBoost Algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)  
  [Use XGBoost with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html)  
  [Perform Automatic Model Tuning with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)  
  [HyperparameterTuner](https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html)  
  [Analyze Results of a Hyperparameter Tuning Job](https://github.com/aws/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb)
  [Amazon SageMaker Model Building Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)   
  [SageMaker Pipelines](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html)  
  [Orchestrating Jobs with Amazon SageMaker Model Building Pipelines](https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-pipelines/tabular/abalone_build_train_deploy/sagemaker-pipelines-preprocess-train-evaluate-batch-transform.ipynb)  
  [Prebuilt Amazon SageMaker Docker Images for Scikit-Learn and Spark ML](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-docker-containers-scikit-learn-spark.html)  
  [Extend a Prebuilt Container](https://docs.aws.amazon.com/sagemaker/latest/dg/prebuilt-containers-extend.html)   
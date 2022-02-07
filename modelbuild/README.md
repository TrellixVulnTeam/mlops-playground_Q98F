## Layout of the SageMaker ModelBuild Project Template

The template provides a starting point for bringing your SageMaker Pipeline development to production.

```
|-- codebuild-buildspec.yml
|-- conf
|   |-- __init__.py
|   `-- config.ini
|-- pipelines
|   |-- ieee_fraud_detection
|   |   |-- inference
|   |   |   |-- sklearn
|   |   |   |   |-- __init__.py
|   |   |   |   `-- inference.py
|   |   |   `-- xgboost
|   |   |       |-- __init__.py
|   |   |       `-- inference.py
|   |   |-- __init__.py
|   |   |-- evaluate.py
|   |   |-- pipeline.py
|   |   |-- preprocess.py
|   |   `-- re_preprocess.py
|   |-- __init__.py
|   |-- __version__.py
|   |-- _utils.py
|   |-- get_pipeline_definition.py
|   `-- run_pipeline.py
|-- README.md
|-- LICENSE
|-- setup.cfg
|-- setup.py
|-- tests
|   `-- test_pipelines.py
`-- tox.ini
```

## Start here
This is a sample code repository that demonstrates how you can organize your code for an ML business solution. This code repository is created as part of creating a Project in SageMaker. 

In this example, we are solving [IEEE-CIS fraud detection](https://www.kaggle.com/c/ieee-fraud-detection) problem using the Kaggle datasets. The following section provides an overview of how the code is organized and what you need to modify. In particular, `pipelines\pipelines.py` contains the core of the business logic for this problem. It has the code to express the ML steps involved in generating an ML model. You will also find the code for that supports preprocessing and evaluation steps in `preprocess.py` and `evaluate.py` files respectively.

Once you understand the code structure described below, you can inspect the code, and you can start customizing it for your own business case. This is only sample code, and you own this repository for your business use case. Please go ahead, modify the files, commit them and see the changes kick off the SageMaker pipelines in the CI/CD system.

A description of some of the artifacts is provided below:
<br/><br/>
Your codebuild execution instructions. This file contains the instructions needed to kick off an execution of the SageMaker Pipeline in the CI/CD system (via CodePipeline). You will see that this file has the fields defined for naming the Pipeline, ModelPackageGroup etc. You can customize them as required.

```
|-- codebuild-buildspec.yml
```

<br/><br/>
Your pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the AUROC of the model that's trained by the pipeline. This is the core business logic, and if you want to create your own folder, you can do so, and implement the `get_pipeline` interface as illustrated here.

```
|-- pipelines
|   |-- ieee_fraud_detection
|   |   |-- inference
|   |   |   |-- sklearn
|   |   |   |   |-- __init__.py
|   |   |   |   `-- inference.py
|   |   |   `-- xgboost
|   |   |       |-- __init__.py
|   |   |       `-- inference.py
|   |   |-- __init__.py
|   |   |-- evaluate.py
|   |   |-- pipeline.py
|   |   |-- preprocess.py
|   |   `-- re_preprocess.py

```
<br/><br/>
Utility modules for getting pipeline definition jsons and running pipelines (you do not typically need to modify these):

```
|-- pipelines
|   |-- __init__.py
|   |-- __version__.py
|   |-- _utils.py
|   |-- get_pipeline_definition.py
|   `-- run_pipeline.py
```
<br/><br/>
Python package artifacts:
```
|-- setup.cfg
|-- setup.py
```
<br/><br/>
A stubbed testing module for testing your pipeline as you develop:
```
|-- tests
|   `-- test_pipelines.py
```
<br/><br/>
The `tox` testing framework configuration:
```
`-- tox.ini
```
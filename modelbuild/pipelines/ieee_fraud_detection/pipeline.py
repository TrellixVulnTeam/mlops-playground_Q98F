import configparser
import os
import boto3
import sagemaker
from sagemaker import PipelineModel
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnModel
from sagemaker.sklearn.processing import ScriptProcessor

# from sagemaker.tuner import (
#     ContinuousParameter,
#     IntegerParameter,
#     HyperparameterTuner,
# )
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
    ConditionLessThanOrEqualTo,
)
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import (
    CacheConfig,
    ProcessingStep,
    TrainingStep,
    # TuningStep,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.xgboost import XGBoostModel

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_sagemaker_client(region):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    sagemaker_runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=sagemaker_runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)

    except Exception as e:
        print(f"Error getting project tags: {e}")

    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name=None,
    pipeline_name=None,
    base_job_prefix=None,
):
    config = configparser.ConfigParser()
    config_path = os.path.join(
        os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-3]),
        "conf",
        "config.ini",
    )
    _ = config.read(config_path)

    default_bucket = (
        config["proj"]["s3_default_bucket"]
        if default_bucket is None
        else default_bucket
    )
    base_job_prefix = (
        config["proj"]["s3_base_job_prefix"]
        if base_job_prefix is None
        else base_job_prefix
    )
    repository_name = config["proj"]["ecr_repository_name"]
    # tuning_max_jobs = eval(config["model"]["tuning_max_jobs"])
    valid_size = config["model"]["valid_size"]
    test_size = config["model"]["test_size"]
    model_package_group_name = (
        config["pipelines"]["model_package_group_name"]
        if model_package_group_name is None
        else model_package_group_name
    )
    pipeline_name = (
        config["pipelines"]["pipeline_name"] if pipeline_name is None else pipeline_name
    )
    target_key = config["pipelines"]["target_key"]
    target_value = eval(config["pipelines"]["target_value"])
    minimize_target = eval(config["pipelines"]["minimize_target"])

    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    boto_session = boto3.Session(region_name=region)
    account_id = boto_session.client("sts").get_caller_identity().get("Account")

    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.2xlarge"
    )
    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount", default_value=1
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.2xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    training_data_uri = f"s3://{config['proj']['s3_default_bucket']}/{config['proj']['s3_base_job_prefix']}/raw_data/training"
    training_data = ParameterString(
        name="TrainingData", default_value=training_data_uri
    )

    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")

    processing_image_uri = (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}"
    )

    preprocessor = ScriptProcessor(
        role=role,
        image_uri=processing_image_uri,
        command=["python3"],
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        base_job_name=f"{base_job_prefix}-data-prep",
    )

    step_preprocess = ProcessingStep(
        name="PreprocessData",
        processor=preprocessor,
        inputs=[
            ProcessingInput(
                source=training_data, destination="/opt/ml/processing/raw_data/training"
            )
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/train", output_name="train"),
            ProcessingOutput(source="/opt/ml/processing/valid", output_name="valid"),
            ProcessingOutput(source="/opt/ml/processing/test", output_name="test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        cache_config=cache_config,
        job_arguments=[
            "--base_dir",
            "/opt/ml/processing",
            "--valid_size",
            valid_size,
            "--test_size",
            test_size,
            "--is_prediction",
            "False",
        ],
    )

    training_image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.3-1",
        py_version="py3",
        instance_type="ml.m5.2xlarge",
    )
    model_output_uri = f"{default_bucket}/{base_job_prefix}/models"

    estimator = Estimator(
        image_uri=training_image_uri,
        role=role,
        instance_count=training_instance_count,
        instance_type=training_instance_type,
        output_path="s3://" + model_output_uri,
    )

    hyperparameters = {
        "booster": "gbtree",
        "verbosity": 0,
        "objective": "binary:logistic",
        "seed": 42,
        "scale_pos_weight": 1.0,
        "eval_metric": "auc",
        "num_round": 1000,
        "early_stopping_rounds": 10,
    }
    estimator.set_hyperparameters(**hyperparameters)

    # hyperparameter_ranges = {
    #     "max_depth": IntegerParameter(1, 30, scaling_type="Auto"),
    #     "eta": ContinuousParameter(0.01, 1.0, scaling_type="Auto"),
    #     "gamma": ContinuousParameter(0.0, 1.0, scaling_type="Auto"),
    #     "min_child_weight": ContinuousParameter(1e-6, 1.0, scaling_type="Auto"),
    #     "subsample": ContinuousParameter(0.1, 1.0, scaling_type="Auto"),
    #     "colsample_bytree": ContinuousParameter(0.1, 1.0, scaling_type="Auto"),
    # }

    # tuner = HyperparameterTuner(
    #     estimator,
    #     "validation:auc",
    #     hyperparameter_ranges,
    #     objective_type="Maximize",
    #     max_jobs=tuning_max_jobs,
    #     max_parallel_jobs=3,
    #     base_tuning_job_name=f"{base_job_prefix}-param-tuning",
    #     early_stopping_type="Auto",
    # )

    # step_tune = TuningStep(
    #     name="TuneHyperparameters",
    #     tuner=tuner,
    #     inputs={
    #         "train": TrainingInput(
    #             s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
    #                 "train"
    #             ].S3Output.S3Uri,
    #             content_type="text/csv",
    #         ),
    #         "validation": TrainingInput(
    #             s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
    #                 "valid"
    #             ].S3Output.S3Uri,
    #             content_type="text/csv",
    #         ),
    #     },
    #     cache_config=cache_config,
    # )

    step_train = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "valid"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )

    evaluator = ScriptProcessor(
        role=role,
        image_uri=training_image_uri,
        command=["python3"],
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        base_job_name=f"{base_job_prefix}-model-eval",
    )

    evaluation = PropertyFile(
        name="ModelEvaluation", output_name="evaluation", path="eval_metrics.json"
    )
    model_data = step_train.properties.ModelArtifacts.S3ModelArtifacts
    # model_data = step_tune.get_top_model_s3_uri(top_k=0, s3_bucket=model_output_uri)

    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=evaluator,
        inputs=[
            ProcessingInput(
                source=model_data,
                destination="/opt/ml/processing/models",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/eval", output_name="evaluation")
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation],
        cache_config=cache_config,
        job_arguments=[
            "--base_dir",
            "/opt/ml/processing",
        ],
    )

    step_re_preprocess = ProcessingStep(
        name="Re-preprocessData",
        processor=preprocessor,
        inputs=[
            ProcessingInput(
                source=training_data, destination="/opt/ml/processing/raw_data/training"
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/re_train", output_name="re_train"
            ),
            ProcessingOutput(
                source="/opt/ml/processing/re_test", output_name="re_test"
            ),
            ProcessingOutput(
                source="/opt/ml/processing/model", output_name="preprocessor"
            ),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        cache_config=cache_config,
        job_arguments=[
            "--base_dir",
            "/opt/ml/processing",
            "--test_size",
            test_size,
            "--is_prediction",
            "True",
        ],
    )

    step_re_train = TrainingStep(
        name="Re-trainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_re_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "re_train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_re_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "re_test"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            content_type="application/json",
            s3_uri=f"{step_evaluate.arguments['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']}/\
            eval_metrics.json",
        ),
    )

    sklearn_model = SKLearnModel(
        model_data=step_re_preprocess.properties.ProcessingOutputConfig.Outputs[
            "preprocessor"
        ].S3Output.S3Uri,
        role=role,
        entry_point="inference.py",
        source_dir=os.path.join(BASE_DIR, "inference", "sklearn") + os.path.sep,
        name="SKLearn",
        code_location=f"s3://{default_bucket}/{base_job_prefix}/sklearn/code",
        image_uri=processing_image_uri,
        sagemaker_session=sagemaker_session,
    )

    xgboost_model = XGBoostModel(
        model_data=step_re_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point="inference.py",
        source_dir=os.path.join(BASE_DIR, "inference", "xgboost") + os.path.sep,
        py_version="py3",
        framework_version="1.3-1",
        name="XGBoost",
        code_location=f"s3://{default_bucket}/{base_job_prefix}/xgboost/code",
        sagemaker_session=sagemaker_session,
    )

    pipeline_model = PipelineModel(
        models=[sklearn_model, xgboost_model],
        role=role,
        sagemaker_session=sagemaker_session,
    )

    step_register = RegisterModel(
        name="RegisterModel",
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.2xlarge"],
        transform_instances=["ml.m5.2xlarge"],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        approval_status=model_approval_status,
        model=pipeline_model,
    )

    step = (
        ConditionLessThanOrEqualTo if minimize_target else ConditionGreaterThanOrEqualTo
    )
    condition = step(
        left=JsonGet(
            step_name="EvaluateModel",
            property_file=evaluation,
            json_path=f"binary_classification_metrics.{target_key}.value",
        ),
        right=target_value,
    )

    step_check = ConditionStep(
        name="CheckCondition",
        conditions=[condition],
        if_steps=[step_re_preprocess, step_re_train, step_register],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            processing_instance_type,
            training_instance_count,
            training_instance_type,
            training_data,
            model_approval_status,
        ],
        steps=[step_preprocess, step_train, step_evaluate, step_check],
        # steps=[step_preprocess, step_tune, step_evaluate, step_check],
        sagemaker_session=sagemaker_session,
    )

    return pipeline

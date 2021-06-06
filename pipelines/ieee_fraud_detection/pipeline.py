import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import ScriptProcessor, SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
    ConditionLessThanOrEqualTo,
)
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))


def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    model_package_group_name,
    pipeline_name,
    base_job_prefix,
    role=None,
    default_bucket=None,
    base_dir="/opt/ml/processing",
):
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    training_data_uri = f"s3://{default_bucket}/{base_job_prefix}/training"
    prediction_data_uri = f"s3://{default_bucket}/{base_job_prefix}/prediction"

    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.2xlarge"
    )
    training_data = ParameterString(
        name="TrainingData", default_value=training_data_uri
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.2xlarge"
    )
    prediction_data = ParameterString(
        name="PredictionData",
        default_value=prediction_data_uri,
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")

    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}-sklearn-processing",
    )

    step_preprocess = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=training_data, destination=base_dir + "/training")
        ],
        outputs=[
            ProcessingOutput(source=base_dir + "/train", output_name="train"),
            ProcessingOutput(source=base_dir + "/valid", output_name="valid"),
            ProcessingOutput(source=base_dir + "/test", output_name="test"),
        ],
        code=os.path.join(WORKING_DIR, "preprocessing.py"),
        cache_config=cache_config,
        job_arguments=["--base_dir", base_dir],
    )

    model_output_uri = f"s3://{default_bucket}/{base_job_prefix}/models"
    training_image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    estimator = Estimator(
        image_uri=training_image_uri,
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=model_output_uri,
        use_spot_instances=False,
        max_wait=None,
    )
    params = {
        "booster": "gbtree",
        "verbosity": 0,
        "objective": "binary:logistic",
        "seed": 42,
        "max_depth": 6,
        "eta": 0.3,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "scale_pos_weight": 1.0,
        "eval_metric": "auc",
        "num_round": 1000,
        "early_stopping_rounds": 10,
    }
    estimator.set_hyperparameters(**params)

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

    script_processor = ScriptProcessor(
        role=role,
        image_uri=training_image_uri,
        command=["python3"],
        instance_count=1,
        instance_type=processing_instance_type,
        base_job_name=f"{base_job_prefix}-script-processing",
    )
    evaluation = PropertyFile(
        name="ModelEvaluation", output_name="evaluation", path="eval_metrics.json"
    )

    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=script_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination=base_dir + "/models",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination=base_dir + "/test",
            ),
        ],
        outputs=[ProcessingOutput(source=base_dir + "/eval", output_name="evaluation")],
        code=os.path.join(WORKING_DIR, "evaluation.py"),
        property_files=[evaluation],
        cache_config=cache_config,
        job_arguments=[
            "--base_dir",
            base_dir,
        ],
    )

    sklearn_re_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}-sklearn-re-processing",
    )

    step_re_preprocess = ProcessingStep(
        name="Re-preprocessData",
        processor=sklearn_re_processor,
        inputs=[
            ProcessingInput(source=training_data, destination=base_dir + "/training"),
            ProcessingInput(
                source=prediction_data, destination=base_dir + "/prediction"
            ),
        ],
        outputs=[
            ProcessingOutput(source=base_dir + "/re_train", output_name="re_train"),
            ProcessingOutput(source=base_dir + "/re_valid", output_name="re_valid"),
            ProcessingOutput(source=base_dir + "/re_test", output_name="re_test"),
        ],
        code=os.path.join(WORKING_DIR, "re_preprocessing.py"),
        cache_config=cache_config,
        job_arguments=[
            "--base_dir",
            base_dir,
        ],
    )

    full_estimator = Estimator(
        image_uri=training_image_uri,
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=model_output_uri,
        use_spot_instances=False,
        max_wait=None,
    )
    full_estimator.set_hyperparameters(**params)

    step_re_train = TrainingStep(
        name="Re-trainModel",
        estimator=full_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_re_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "re_train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_re_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "re_valid"
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

    step_register = RegisterModel(
        name="RegisterModel",
        estimator=full_estimator,
        model_data=step_re_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.2xlarge"],
        transform_instances=["ml.m5.2xlarge"],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        approval_status=model_approval_status,
    )

    target_metric = "auroc"
    target_value = 0.9
    target_minimize = False

    step = (
        ConditionLessThanOrEqualTo if target_minimize else ConditionGreaterThanOrEqualTo
    )
    condition = step(
        left=JsonGet(
            step=step_evaluate,
            property_file=evaluation,
            json_path=f"eval_metric.{target_metric}",
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
            training_data,
            training_instance_type,
            prediction_data,
            model_approval_status,
        ],
        steps=[step_preprocess, step_train, step_evaluate, step_check],
    )

    return pipeline

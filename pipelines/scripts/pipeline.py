import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import ScriptProcessor, SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionLessThanOrEqualTo
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_session(region, bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client('sagemaker')
    runtime_client = boto_session.client('sagemaker-runtime')
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=bucket
    )


def get_pipeline(region, base_job_prefix, model_package_group_name, pipeline_name, bucket=None, role=None):
    sagemaker_session = get_session(region, bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    training_data_uri = f's3://{bucket}/{base_job_prefix}/raw_data/training'
    prediction_data_uri = f's3://{bucket}/{base_job_prefix}/raw_data/prediction'

    processing_instance_count = ParameterInteger(
        name='ProcessingInstanceCount',
        default_value=1
    )
    processing_instance_type = ParameterString(
        name='ProcessingInstanceType',
        default_value='ml.m5.2xlarge'
    )
    training_data = ParameterString(
        name='TrainingData',
        default_value=training_data_uri
    )
    training_instance_type = ParameterString(
        name='TrainingInstanceType',
        default_value='ml.m5.2xlarge'
    )
    prediction_data = ParameterString(
        name='PredictionData',
        default_value=prediction_data_uri,
    )
    model_approval_status = ParameterString(
        name='ModelApprovalStatus',
        default_value='PendingManualApproval'
    )
    
    sklearn_processor = SKLearnProcessor(
        framework_version='0.23-1',
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f'{base_job_prefix}-sklearn-processing'
    )

    step_preprocess = ProcessingStep(
        name='PreprocessData',
        processor=sklearn_processor,
        inputs=[
          ProcessingInput(source=training_data, destination='/opt/ml/processing/training') 
        ],
        outputs=[
            ProcessingOutput(source='/opt/ml/processing/train', output_name='train'),
            ProcessingOutput(source='/opt/ml/processing/valid', output_name='valid'),
            ProcessingOutput(source='/opt/ml/processing/test', output_name='test')
        ],
        code=os.path.join(BASE_DIR, 'preprocessing.py')
    )

    model_output_uri = f's3://{bucket}/{base_job_prefix}/models'
    image_uri = sagemaker.image_uris.retrieve(
        framework='xgboost',
        region=region,
        version='1.2-1',
        py_version='py3',
        instance_type=training_instance_type
    )

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=model_output_uri,
        use_spot_instances=False,
        max_wait=None
    )
    params = {
        'booster': 'gbtree',
        'verbosity': 0,
        'objective': 'binary:logistic',
        'seed': 42,
        'max_depth': 6,
        'eta': 0.3,
        'gamma': 0.0,
        'min_child_weight': 1.0,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'scale_pos_weight': 1.0,
        'eval_metric': 'auc',
        'num_round': 1000,
        'early_stopping_rounds': 10
    }
    estimator.set_hyperparameters(**params)

    step_train = TrainingStep(
        name='TrainModel',
        estimator=estimator,
        inputs={
            'train': TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri,
                content_type='text/csv'
            ),
            'validation': TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs['valid'].S3Output.S3Uri,
                content_type='text/csv'
            )
        }
    )

    script_processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=['python3'],
        instance_count=1,
        instance_type=processing_instance_type,
        base_job_name=f'{base_job_prefix}-script-processing'
    )
    evaluation = PropertyFile(
        name='ModelEvaluation',
        output_name='evaluation',
        path='eval_metrics.json'
    )

    step_evaluate = ProcessingStep(
        name='EvaluateModel',
        processor=script_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination='/opt/ml/processing/models'
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri,
                destination='/opt/ml/processing/test'
            )
        ],
        outputs=[
            ProcessingOutput(source='/opt/ml/processing/eval', output_name='evaluation')
        ],
        code=os.path.join(BASE_DIR, 'evaluation.py'),
        property_files=[evaluation]
    )

    sklearn_reprocessor = SKLearnProcessor(
        framework_version='0.23-1',
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f'{base_job_prefix}-sklearn-reprocessing'
    )

    step_repreprocess = ProcessingStep(
        name='RepreprocessData',
        processor=sklearn_reprocessor,
        inputs=[
            ProcessingInput(source=training_data, destination='/opt/ml/processing/training'),
            ProcessingInput(source=prediction_data, destination='/opt/ml/processing/prediction')
        ],
        outputs=[
            ProcessingOutput(source='/opt/ml/processing/retrain', output_name='retrain'),
            ProcessingOutput(source='/opt/ml/processing/revalid', output_name='revalid'),
            ProcessingOutput(source='/opt/ml/processing/retest', output_name='retest')
        ],
        code=os.path.join(BASE_DIR, 'repreprocessing.py')
    )

    full_estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=model_output_uri,
        use_spot_instances=False,
        max_wait=None
    )
    full_estimator.set_hyperparameters(**params)

    step_retrain = TrainingStep(
        name='RetrainModel',
        estimator=full_estimator,
        inputs={
            'train': TrainingInput(
                s3_data=step_repreprocess.properties.ProcessingOutputConfig.Outputs['retrain'].S3Output.S3Uri,
                content_type='text/csv'
            ),
            'validation': TrainingInput(
                s3_data=step_repreprocess.properties.ProcessingOutputConfig.Outputs['revalid'].S3Output.S3Uri,
                content_type='text/csv'
            )
        }
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            content_type='application/json',
            s3_uri='{}/eval_metrics.json'.format(
                step_evaluate.arguments['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']
            )
        )
    )

    step_register = RegisterModel(
        name='RegisterModel',
        estimator=full_estimator,
        model_data=step_retrain.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=['text/csv'],
        response_types=['text/csv'],
        inference_instances=['ml.t2.medium', 'ml.m5.2xlarge'],
        transform_instances=['ml.m5.2xlarge'],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        approval_status=model_approval_status
    )

    target_metric = 'auroc'
    target_value = 0.9
    target_minimize = False

    step = ConditionLessThanOrEqualTo if target_minimize else ConditionGreaterThanOrEqualTo
    condition = step(
        left=JsonGet(
            step=step_evaluate,
            property_file=evaluation,
            json_path=f'eval_metric.{target_metric}'
        ),
        right=target_value
    )

    step_check = ConditionStep(
        name='CheckCondition',
        conditions=[condition],
        if_steps=[step_repreprocess, step_retrain, step_register],
        else_steps=[]
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            processing_instance_type,
            training_data,
            training_instance_type,
            prediction_data,
            model_approval_status
        ],
        steps=[step_preprocess, step_train, step_evaluate, step_check]
    )

    return pipeline

import configparser
import os
import shutil
import boto3
import sagemaker
from sagemaker import s3
from utils.common import find_pattern
from botocore.client import ClientError


SEED_CODE_PATH = os.path.join("..", "seed_code")
FILE_NAMES_TO_IGNORE = [".DS_Store"]


if __name__ == "__main__":
    config = configparser.ConfigParser()
    _ = config.read("config.ini")

    raw_data_dir = config["proj"]["ebs_raw_data_dir"]
    default_bucket = config["proj"]["s3_default_bucket"]
    base_job_prefix = config["proj"]["s3_base_job_prefix"]

    sagemaker_session = sagemaker.session.Session()
    if len(default_bucket) == 0:
        default_bucket = sagemaker_session.default_bucket()
    boto_session = boto3.Session()
    s3_client = boto_session.client("s3")

    try:
        _ = s3_client.head_bucket(Bucket=default_bucket)
    except ClientError:
        _ = s3_client.create_bucket(Bucket=default_bucket)

    file_names = [
        "train_identity.csv",
        "train_transaction.csv",
        "test_identity.csv",
        "test_transaction.csv",
    ]
    prefixes = ["training"] * 2 + ["prediction"] * 2

    for file_name, prefix in zip(file_names, prefixes):
        s3.S3Uploader().upload(
            os.path.join(raw_data_dir, file_name),
            f"s3://{default_bucket}/{base_job_prefix}/raw_data/{prefix}",
            sagemaker_session=sagemaker_session,
        )

    print("The raw data has been successfully uploaded to the S3 bucket.")

    os.makedirs(SEED_CODE_PATH, exist_ok=True)
    dir_names = [os.path.join("..", "modelbuild"), os.path.join("..", "modeldeploy")]
    file_names = ["model-building-workflow-v1.0", "mpg-deployment-config-v1.0"]

    for dir_name in dir_names:
        for file_name in FILE_NAMES_TO_IGNORE:
            for result in find_pattern(f"*{file_name}", dir_name):
                os.remove(result)

    for dir_name, file_name in zip(dir_names, file_names):
        shutil.make_archive(
            os.path.join(SEED_CODE_PATH, file_name),
            "zip",
            dir_name,
        )

    for file_name in file_names:
        s3.S3Uploader().upload(
            os.path.join(SEED_CODE_PATH, f"{file_name}.zip"),
            f"s3://{default_bucket}/{base_job_prefix}/seed-code",
            sagemaker_session=sagemaker_session,
        )

    print(
        "The seed codes have been compressed and uploaded to the S3 bucket successfully."
    )

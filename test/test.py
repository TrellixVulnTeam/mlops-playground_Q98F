import configparser
import os
import warnings
import sagemaker
import numpy as np
import pandas as pd
from sagemaker.serializers import CSVSerializer

warnings.filterwarnings(action="ignore")
np.random.seed(42)

ENDPOINT_NAME = "ieee-fraud-detection"

if __name__ == "__main__":
    config = configparser.ConfigParser()
    _ = config.read(os.path.join("..", "conf", "config.ini"))

    raw_data_dir = config["proj"]["ebs_raw_data_dir"]
    sagemaker_session = sagemaker.Session()

    test_identity = pd.read_csv(f"{raw_data_dir}/test_identity.csv")
    test_transaction = pd.read_csv(f"{raw_data_dir}/test_transaction.csv")
    df_test = pd.merge(test_transaction, test_identity, on="TransactionID", how="left")
    df_test = df_test.rename(
        columns={"id-{:02d}".format(i): "id_{:02d}".format(i) for i in range(1, 39)}
    )

    cat_features = pd.Index(
        [
            "ProductCD",
            "addr1",
            "addr2",
            "P_emaildomain",
            "R_emaildomain",
            "DeviceType",
            "DeviceInfo",
        ]
        + [f"card{i}" for i in range(1, 7)]
        + [f"M{i}" for i in range(1, 10)]
        + [f"id_{i}" for i in range(12, 39)]
    )
    num_features = df_test.columns.difference(
        pd.Index(["TransactionID", "TransactionDT"]) | cat_features
    )
    all_features = (cat_features | num_features).sort_values()

    n_samples = 10
    df_samples = df_test.sample(n_samples)

    predictor = sagemaker.predictor.Predictor(
        endpoint_name=f"{ENDPOINT_NAME}-prod",
        serializer=CSVSerializer(),
        sagemaker_session=sagemaker_session,
    )

    for _, row in df_samples.iterrows():
        response = predictor.predict(row[all_features].values)
        score = eval(response)[0]
        print(
            f"The predictive score for fraud with transaction ID {row['TransactionID']} is {round(score, 6)}."
        )

    print("The endpoint test completed successfully.")

import joblib
import os
import pandas as pd
from io import StringIO


FEATURE_NAMES = (
    [
        "ProductCD",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "TransactionAmt",
        "dist1",
        "dist2",
    ]
    + [f"card{i}" for i in range(1, 7)]
    + [f"M{i}" for i in range(1, 10)]
    + [f"id_{i:02d}" for i in range(1, 39)]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"V{i}" for i in range(1, 340)]
)


def input_fn(input_data, content_type):
    if content_type == "text/csv":
        df = pd.read_csv(StringIO(input_data), header=None, index_col=False, sep=",")
        first_row = df.iloc[0:1].values[0].tolist()
        if len(df.columns) == len(FEATURE_NAMES):
            print("The column length is correct.")
            if set(first_row) == set(FEATURE_NAMES):
                print("Since the rows contain a header, the first row is removed.")
                df = df.iloc[1:]
                df.reset_index(drop=True, inplace=True)
            df.columns = sorted(FEATURE_NAMES)

        return df
    else:
        raise ValueError(f"{content_type} is not supported in this script.")


def predict_fn(input_data, model):
    input_data.head(1)
    features = model.transform(input_data)
    print("The sklearn inference was successful as follows: ", features)
    return features


def model_fn(model_dir):
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor

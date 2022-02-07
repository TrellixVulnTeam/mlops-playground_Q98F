import argparse
import joblib
import tarfile
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder


def str_to_int(x):
    return x if pd.isnull(x) else str(int(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/opt/ml/processing", type=str)
    parser.add_argument("--valid_size", default=0.15, type=float)
    parser.add_argument("--test_size", default=0.15, type=float)
    parser.add_argument("--is_prediction", default="False", type=str)

    args = parser.parse_args()
    base_dir = args.base_dir
    valid_size = args.valid_size
    test_size = args.test_size
    is_prediction = eval(args.is_prediction)

    train_identity = pd.read_csv(f"{base_dir}/raw_data/training/train_identity.csv")
    train_transaction = pd.read_csv(
        f"{base_dir}/raw_data/training/train_transaction.csv"
    )
    df_train = pd.merge(
        train_transaction, train_identity, on="TransactionID", how="left"
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
    num_features = df_train.columns.difference(
        pd.Index(["TransactionID", "TransactionDT", "isFraud"]) | cat_features
    )
    all_features = (cat_features | num_features).sort_values()

    int_cat_features = df_train[cat_features].select_dtypes("number").columns
    df_train[int_cat_features] = df_train[int_cat_features].applymap(str_to_int)
    df_train[cat_features] = df_train[cat_features].astype("str")

    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
        df_train[all_features],
        df_train["isFraud"],
        test_size=test_size,
        random_state=42,
        stratify=df_train["isFraud"],
    )

    if not is_prediction:
        df_X_train, df_X_valid, df_y_train, df_y_valid = train_test_split(
            df_X_train,
            df_y_train,
            test_size=valid_size / (1.0 - test_size),
            random_state=42,
            stratify=df_y_train,
        )

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="<unknown>"),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        TargetEncoder(drop_invariant=True, min_samples_leaf=1, smoothing=1.0),
    )
    num_pipeline = SimpleImputer(strategy="median")
    preprocessor = make_column_transformer(
        (cat_pipeline, cat_features), (num_pipeline, num_features)
    )

    X_train = preprocessor.fit_transform(df_X_train, df_y_train)
    arr_train = np.concatenate((df_y_train.values.reshape(-1, 1), X_train), axis=1)

    if is_prediction:
        joblib.dump(preprocessor, f"{base_dir}/model/model.joblib")
        tar = tarfile.open(f"{base_dir}/model/model.tar.gz", "w:gz")
        tar.add(f"{base_dir}/model/model.joblib", arcname="model.joblib")
        tar.close()

    else:
        X_valid = preprocessor.transform(df_X_valid)
        arr_valid = np.concatenate((df_y_valid.values.reshape(-1, 1), X_valid), axis=1)

    X_test = preprocessor.transform(df_X_test)
    arr_test = np.concatenate((df_y_test.values.reshape(-1, 1), X_test), axis=1)

    dir_names = ["re_train", "re_test"] if is_prediction else ["train", "valid", "test"]
    file_names = (
        ["arr_re_train", "arr_re_test"]
        if is_prediction
        else ["arr_train", "arr_valid", "arr_test"]
    )

    for dir_name, file_name, dataset in zip(
        dir_names,
        file_names,
        [arr_train, arr_test] if is_prediction else [arr_train, arr_valid, arr_test],
    ):
        np.savetxt(
            f"{base_dir}/{dir_name}/{file_name}.csv",
            dataset,
            delimiter=",",
            fmt="%i",
        )

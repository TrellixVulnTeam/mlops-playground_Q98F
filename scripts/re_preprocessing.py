import argparse
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
    parser.add_argument("--test_size", default=0.25, type=float)

    args = parser.parse_args()
    base_dir = args.base_dir
    test_size = args.test_size

    train_identity = pd.read_csv(f"{base_dir}/training/train_identity.csv")
    train_transaction = pd.read_csv(f"{base_dir}/training/train_transaction.csv")
    df_train = pd.merge(
        train_transaction, train_identity, on="TransactionID", how="left"
    )

    test_identity = pd.read_csv(f"{base_dir}/prediction/test_identity.csv")
    test_transaction = pd.read_csv(f"{base_dir}/prediction/test_transaction.csv")
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
    num_features = df_train.columns.difference(
        pd.Index(["TransactionID", "TransactionDT", "isFraud"]) | cat_features
    )
    all_features = cat_features | num_features

    int_cat_features = df_train[cat_features].select_dtypes("number").columns
    df_train[int_cat_features] = df_train[int_cat_features].applymap(str_to_int)
    df_train[cat_features] = df_train[cat_features].astype("str")

    df_X_re_train, df_X_re_valid, df_y_re_train, df_y_re_valid = train_test_split(
        df_train[all_features],
        df_train["isFraud"],
        test_size=test_size,
        random_state=42,
        stratify=df_train["isFraud"],
    )

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="<unknown>"),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        TargetEncoder(drop_invariant=True, min_samples_leaf=1, smoothing=1.0),
    )
    num_pipeline = SimpleImputer(strategy="median")
    processor = make_column_transformer(
        (cat_pipeline, cat_features), (num_pipeline, num_features)
    )

    X_re_train = processor.fit_transform(df_X_re_train, df_y_re_train)
    X_re_valid = processor.transform(df_X_re_valid)
    X_re_test = processor.transform(df_test[all_features])

    arr_train = np.concatenate(
        (df_y_re_train.values.reshape(-1, 1), X_re_train), axis=1
    )
    arr_valid = np.concatenate(
        (df_y_re_valid.values.reshape(-1, 1), X_re_valid), axis=1
    )
    arr_test = X_re_test

    np.savetxt(f"{base_dir}/re_train/arr_train.csv", arr_train, delimiter=",", fmt="%i")
    np.savetxt(f"{base_dir}/re_valid/arr_valid.csv", arr_valid, delimiter=",", fmt="%i")
    np.savetxt(f"{base_dir}/re_test/arr_test.csv", arr_test, delimiter=",", fmt="%i")

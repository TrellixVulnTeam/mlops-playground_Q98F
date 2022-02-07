import json
import numpy as np
import xgboost as xgb
from sagemaker_inference import content_types, encoder


def input_fn(input_data, content_type):
    if content_type == content_types.JSON:
        print("The received content type is JSON.")
        print("Input_data is: ", input_data)
        obj = json.loads(input_data)
        print("The object parsed as JSON is as follows: ", obj)
        array = np.array(obj)
        return xgb.DMatrix(array)
    else:
        print("The received content type is not JSON.")
        return encoder.decode(input_data, content_type)


def model_fn(model_dir):
    model = xgb.Booster()
    model.load_model(model_dir + "/xgboost-model")
    return model

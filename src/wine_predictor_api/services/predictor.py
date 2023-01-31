import os
import pickle
from wine_predictor_api import api_config
from wine_predictor_api import logger
import numpy as np


def load_model():
    """Load the machine learning model"""
    output_path = api_config.get("model", {}).get("path", "") + "/my_model"
    if not os.path.exists(output_path) or not os.path.isfile(output_path):
        raise FileNotFoundError("File not found")
    with open(output_path, 'rb') as files:
        return pickle.load(files)


def get_features():
    """Returns the list of all 11 feature names for predicting the quality of the wine"""
    return [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "ph",
        "sulphates",
        "alcohol"
    ]


def prepare_data(data):
    """Transforms all user inputs (11 params) from Dict to Numpy array"""
    features = get_features()
    data = [value for key, value in data.items() if key in features]
    return np.array([data])


def validate_input(input):
    for key in get_features():
        if key not in input:
            return False
    return True


def estimate_wine_quality(**kwargs):
    """Estimate the quality of a wine from a defined set of features"""
    try:
        if not validate_input(kwargs):
            return "Missing/Invalid required parameter", 201
        model = load_model()
        # features = get_features()
        X = prepare_data(kwargs)
        prediction = model.predict(X)
        logger.debug(f"Predicted value: {prediction}")
        prediction = prediction[0]
        return {"estimation": int(round(prediction, 0))}, 200
    except FileNotFoundError:
        return "Model path is not found", 404
    except:
        return "Internal server error", 500

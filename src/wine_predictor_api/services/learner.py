import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import validators
import pickle
from wine_predictor_api import api_config
from wine_predictor_api import logger


def load_data():
    """Loads the CSV dataset"""
    data_path = api_config.get("data", {}).get("path")
    if not validators.url(data_path) and not os.path.isfile(data_path):
        raise FileNotFoundError("data path could not be found")

    return pd.read_csv(data_path)


def evaluate_model(model, test_data, test_target):
    """Computes the accuracy of the model from the test split"""
    y_predicted = model.predict(test_data)
    return mean_squared_error(test_target, y_predicted)


def save_model(model, output_path):
    """Saves the the hyper-parameters of your model"""
    logger.info(f"Saving model in {output_path}...")
    with open(output_path, 'wb') as files:
        pickle.dump(model, files)


def get_saved_model(output_path):
    if not os.path.exists(output_path) or not os.path.isfile(output_path):
        return None
    with open(output_path, 'rb') as files:
        return pickle.load(files)


def train_model():
    """Trains a model from the train dataset, evaluate the model performance and save it if its performance
    is better the the existing model"""
    default_response = "Internal server error"
    responses = {
        200: "New model has been successfully trained but discarded",
        201: "New model has been successfully trained and saved as default",
        404: "Model path is not found",
        500: default_response
    }
    status_code = 201
    update_model = True
    output_path = api_config.get("model", {}).get("path", "") + "/my_model"

    # We load the data and we split it
    try:
        dataset = load_data()
    except FileNotFoundError as error:
        logger.error(f"File not found: {error}")
        status_code = 404
        return responses.get(status_code, default_response), status_code
    Y = dataset['TARGET']
    X = dataset.drop('TARGET', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    # We train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    new_mse = evaluate_model(model, x_test, y_test)
    logger.debug(f"New model mse: {new_mse}")
    old_model = get_saved_model(output_path)

    if old_model:
        old_mse = evaluate_model(old_model, x_test, y_test)
        logger.debug(f"Old model mse: {old_mse}")
        if old_mse <= new_mse:
            status_code = 200
            update_model = False

    if update_model:
        logger.debug(f"New model saved with mse: {new_mse}. (old value {old_mse})")
        save_model(model, output_path)

    return responses.get(status_code, default_response), status_code

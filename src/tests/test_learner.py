import pandas as pd
import pickle


def aux(output_path):
    with open(output_path, 'rb') as files:
        return pickle.load(files)


def test_train_model(app_client, test_auth, test_data_path, test_model_path, mocker):
    # given
    endpoint = "/wine/model"

    mocker.patch("wine_predictor_api.services.learner.load_data", return_value=pd.read_csv(test_data_path), autospec=True)
    mocker.patch("wine_predictor_api.services.learner.get_saved_model", return_value=aux(test_model_path), autospec=True)
    mocker.patch("wine_predictor_api.services.learner.save_model", return_value=None, autospec=True)

    # when
    response = app_client.patch(endpoint, headers={"Authorization": test_auth})

    # then
    assert response.status_code in [200, 201]

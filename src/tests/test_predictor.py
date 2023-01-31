import pickle


def aux(output_path):
    with open(output_path, 'rb') as files:
        return pickle.load(files)


def test_estimate_wine_quality(app_client, test_auth, test_model_path, mocker):
    # given
    endpoint = "/wine/quality"
    url_params = "fixed_acidity=7.4&volatile_acidity=0.7&citric_acid=0.0&residual_sugar=1.9&chlorides=0.076&free_sulfur_dioxide=11&total_sulfur_dioxide=34&density=0.9978&ph=3.51&sulphates=0.56&alcohol=9.4"
    mocker.patch("wine_predictor_api.services.predictor.load_model", return_value=aux(test_model_path), autospec=True)

    # when
    response = app_client.get(f"{endpoint}?{url_params}", headers={"Authorization": test_auth})

    # then
    assert response.status_code == 200

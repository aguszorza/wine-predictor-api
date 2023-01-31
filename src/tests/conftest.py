import pytest
import os
import base64
from tests.assets import where as asset_dir_path


@pytest.fixture()
def app():
    from wine_predictor_api import create_app
    app = create_app()
    app.config.update(
        TESTING=True,
        LOGIN_DISABLED=True
    )
    yield app


@pytest.fixture
def test_auth():
    test_cred = base64.b64encode(b"test_user:test_pass").decode('utf-8')
    return f"Basic {test_cred}"


@pytest.fixture
def test_data_path():
    return asset_dir_path() + "/sample_data.csv"


@pytest.fixture
def test_model_path():
    return asset_dir_path() + "/test_model.jl"


@pytest.fixture()
def app_client(app):
    return app.test_client()


@pytest.fixture()
def app_runner(app, pytest_configure):
    return app.test_cli_runner()


@pytest.hookimpl
def pytest_configure(config):
    os.environ["API_CONFIG"] = "config.template.json"

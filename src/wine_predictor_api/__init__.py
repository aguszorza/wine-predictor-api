import connexion
from wine_predictor_api import specs
import logging.config
import yaml
import os
from typing import Dict, Any
import json


def init_logger(name=None):
    with open(os.getenv('LOGGING_CONFIG', "logging.yaml")) as stream:
        logging_config = yaml.safe_load(stream)
        logging.config.dictConfig(logging_config)
        return logging.getLogger(name)


def init_config():
    with open(os.getenv('API_CONFIG', "config.json")) as stream:
        return json.load(stream)


logger = init_logger()
api_config: Dict[str, Any] = init_config()


def create_app():
    spec_options = api_config.get("spec_options", {})
    spec_options['version'] = os.environ.get("API_VERSION", 'version_not_set')

    app = connexion.FlaskApp(__name__, specification_dir=specs.where())
    app.add_api("openapi_spec.yaml", arguments=spec_options)
    return app.app


__all__ = ['logger', 'api_config']

from wine_predictor_api import api_config

PASSWD = api_config.get("basic_cred", {})


def basic_auth(username, password):
    """
    Basic Auth implementation
    :param username:
    :param password:
    :return:
    """
    if PASSWD.get(username) == password:
        return {"sub": username}

    return None

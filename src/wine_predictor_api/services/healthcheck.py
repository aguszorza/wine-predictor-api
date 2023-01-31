from wine_predictor_api import logger


def ping():
    logger.debug("Writing for accountability ...")
    logger.info("Just an information ...")
    logger.error("An error occurred ...")
    return "pong", 200

import logging
import sys


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(process)d: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger
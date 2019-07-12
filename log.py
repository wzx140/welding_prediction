import logging
import os

log_path = os.path.join(os.getcwd(), 'resource/log.txt')
if not os.path.exists(log_path):
    open(log_path, 'w')

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_path, encoding='utf-8')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def log_debug(message):
    logger.debug(message)


def log_info(message):
    logger.info(message)


def log_error(error):
    logger.error('\n' + error + '\n')


def log_exception(error):
    logger.exception(error)

import logging
import definitions

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(definitions.ROOT_DIR + 'log/log.txt', encoding='utf-8')
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

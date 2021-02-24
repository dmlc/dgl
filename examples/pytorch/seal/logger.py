import logging
import time
import os


def _transform_log_level(str_level):
    if str_level == 'info':
        return logging.INFO
    elif str_level == 'warning':
        return logging.WARNING
    elif str_level == 'critical':
        return logging.CRITICAL
    elif str_level == 'debug':
        return logging.DEBUG
    elif str_level == 'error':
        return logging.ERROR
    else:
        raise KeyError('Log level error')


class LightLogging(object):
    def __init__(self, log_path=None, log_name='lightlog', log_level='debug'):

        log_level = _transform_log_level(log_level)

        if log_path:
            if not log_path.endswith('/'):
                log_path += '/'
            if not os.path.exists(log_path):
                os.mkdir(log_path)

            if log_name.endswith('-') or log_name.endswith('_'):
                log_name = log_path+log_name + time.strftime('%Y-%m-%d-%H:%M', time.localtime(time.time())) + '.log'
            else:
                log_name = log_path+log_name + '_' + time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'

            logging.basicConfig(level=log_level,
                                format="%(asctime)s %(levelname)s: %(message)s",
                                datefmt='%Y-%m-%d-%H:%M',
                                handlers=[
                                    logging.FileHandler(log_name, mode='w'),
                                    logging.StreamHandler()
                                ])
            logging.info('Start Logging')
            logging.info('Log file path: {}'.format(log_name))

        else:
            logging.basicConfig(level=log_level,
                                format="%(asctime)s %(levelname)s: %(message)s",
                                datefmt='%Y-%m-%d-%H:%M',
                                handlers=[
                                    logging.StreamHandler()
                                ])
            logging.info('Start Logging')

    def debug(self, msg):
        logging.debug(msg)

    def info(self, msg):
        logging.info(msg)

    def critical(self, msg):
        logging.critical(msg)

    def warning(self, msg):
        logging.warning(msg)

    def error(self, msg):
        logging.error(msg)


